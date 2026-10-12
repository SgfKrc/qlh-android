package com.qlh.inference.service

import com.qlh.inference.data.SettingsDataStore
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class InferenceModeCoordinatorTest {
    @Test
    fun localSwitchWaitsForWorkerExitAndTailReleaseBeforeFullLoad() = runBlocking {
        val modes = InferenceModeCoordinator()
        var mode = SettingsDataStore.MODE_DISTRIBUTED
        val stopping = CompletableDeferred<Unit>()
        val exited = CompletableDeferred<Unit>()
        val events = mutableListOf<String>()
        modes.exclusive {
            modes.registerWorker(Any()) {
                stopping.complete(Unit)
                exited.await()
                events += "worker exited"
            }
        }
        val switch = async(start = CoroutineStart.UNDISPATCHED) {
            modes.switchMode(SettingsDataStore.MODE_LOCAL,
                persistMode = { mode = it; events += "mode published" },
                releaseModel = { events += "tail released"; Result.success(Unit) })
        }
        withTimeout(5_000) { stopping.await() }
        val load = async(start = CoroutineStart.UNDISPATCHED) {
            modes.local<Unit>({ mode }, { error("switch must release the tail") }) {
                events += "full loaded"
                Result.success(Unit)
            }
        }
        try {
            assertFalse(switch.isCompleted)
            assertFalse(load.isCompleted)
            assertEquals(SettingsDataStore.MODE_DISTRIBUTED, mode)
            exited.complete(Unit)
            withTimeout(5_000) { switch.await(); load.await().getOrThrow() }
            assertEquals(listOf("worker exited", "tail released", "mode published", "full loaded"), events)
        } finally {
            exited.complete(Unit)
        }
    }

    @Test
    fun rejoinWaitsForFullUnloadBeforeRecreatedWorkerCanAdvertise() = runBlocking {
        val modes = InferenceModeCoordinator()
        var mode = SettingsDataStore.MODE_LOCAL
        val unloading = CompletableDeferred<Unit>()
        val released = CompletableDeferred<Unit>()
        val events = mutableListOf<String>()
        val switch = async(start = CoroutineStart.UNDISPATCHED) {
            modes.switchMode(SettingsDataStore.MODE_DISTRIBUTED,
                persistMode = { mode = it; events += "distributed published" },
                releaseModel = {
                    unloading.complete(Unit)
                    released.await()
                    events += "full released"
                    Result.success(Unit)
                })
        }
        withTimeout(5_000) { unloading.await() }
        val restart = async(start = CoroutineStart.UNDISPATCHED) {
            modes.exclusive {
                assertEquals(SettingsDataStore.MODE_DISTRIBUTED, mode)
                assertTrue(modes.prepareWorkerLocked({ mode }, { Result.success(Unit) }))
                modes.registerWorker(Any()) {}
                events += "worker advertised"
            }
        }
        try {
            assertFalse(restart.isCompleted)
            assertEquals(SettingsDataStore.MODE_LOCAL, mode)
            released.complete(Unit)
            withTimeout(5_000) { switch.await(); restart.await() }
            assertEquals(listOf("full released", "distributed published", "worker advertised"), events)
        } finally {
            released.complete(Unit)
        }
    }

    @Test
    fun distributedRejectsExplicitLocalPreloadEvenWithAnActiveWorker() = runBlocking {
        val modes = InferenceModeCoordinator()
        var stopped = false
        modes.exclusive { modes.registerWorker(Any()) { stopped = true } }
        val result = modes.local<Unit>(
            { SettingsDataStore.MODE_DISTRIBUTED },
            { error("must not release a distributed model") },
        ) { error("explicit preload must not run") }
        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull()!!.message!!.contains("本地模式"))
        assertFalse(stopped)
    }

    @Test
    fun localAndFallbackDenyLayerAndFullWorkerWorkDespitePersistedStartup() = runBlocking {
        for (mode in listOf(SettingsDataStore.MODE_LOCAL, SettingsDataStore.MODE_FALLBACK)) {
            val modes = InferenceModeCoordinator()
            modes.exclusive { modes.registerWorker(Any()) {} }
            assertTrue(modes.worker<Unit>({ mode }) { error("stale worker must not load or run") }.isFailure)
        }
        val modes = InferenceModeCoordinator()
        assertTrue(modes.worker<Unit>({ SettingsDataStore.MODE_DISTRIBUTED }) {
            error("stopped worker must not run")
        }.isFailure)
    }

    @Test
    fun recreatedWorkerCannotResumeInLocalOrFallbackMode() = runBlocking {
        val modes = InferenceModeCoordinator()
        for (mode in listOf(SettingsDataStore.MODE_LOCAL, SettingsDataStore.MODE_FALLBACK)) {
            modes.exclusive {
                assertFalse(modes.prepareWorkerLocked({ mode }, {
                    error("mode refusal must precede model release and registration")
                }))
            }
        }
    }

    @Test
    fun recreatedDistributedWorkerReleasesPreviousFullModelBeforeAdvertising() = runBlocking {
        val modes = InferenceModeCoordinator()
        val releasing = CompletableDeferred<Unit>()
        val released = CompletableDeferred<Unit>()
        val events = mutableListOf<String>()
        val startup = async(start = CoroutineStart.UNDISPATCHED) {
            modes.exclusive {
                assertTrue(modes.prepareWorkerLocked({ SettingsDataStore.MODE_DISTRIBUTED }, {
                    releasing.complete(Unit)
                    released.await()
                    events += "previous full model released"
                    Result.success(Unit)
                }))
                modes.registerWorker(Any()) {}
                events += "worker advertised"
            }
        }
        withTimeout(5_000) { releasing.await() }
        try {
            assertFalse(startup.isCompleted)
            assertTrue(events.isEmpty())
            released.complete(Unit)
            withTimeout(5_000) { startup.await() }
            assertEquals(listOf("previous full model released", "worker advertised"), events)
        } finally {
            released.complete(Unit)
        }
    }

    @Test
    fun backgroundLocalLoadDrainsAnOldWorkerBeforeReusingTheEngine() = runBlocking {
        val modes = InferenceModeCoordinator()
        val events = mutableListOf<String>()
        modes.exclusive { modes.registerWorker(Any()) { events += "worker stopped" } }
        modes.local<Unit>({ SettingsDataStore.MODE_LOCAL }, {
            events += "worker model released"
            Result.success(Unit)
        }) {
            events += "local loaded"
            Result.success(Unit)
        }.getOrThrow()
        assertEquals(listOf("worker stopped", "worker model released", "local loaded"), events)
    }

    @Test
    fun unloadFailureKeepsTheOldModeAndBlocksRejoin() = runBlocking {
        val modes = InferenceModeCoordinator()
        val failure = IllegalStateException("native release failed")
        var published = false
        val result = runCatching {
            modes.switchMode(SettingsDataStore.MODE_DISTRIBUTED,
                persistMode = { published = true },
                releaseModel = { Result.failure(failure) })
        }
        assertEquals(failure.javaClass, result.exceptionOrNull()!!.javaClass)
        assertEquals(failure.message, result.exceptionOrNull()!!.message)
        assertFalse(published)
        assertTrue(modes.worker<Unit>({ SettingsDataStore.MODE_LOCAL }) {
            error("release failure must not admit a worker")
        }.isFailure)
    }

    @Test
    fun cancellationDuringSwitchStillFinishesReleaseAndModePublication() = runBlocking {
        val modes = InferenceModeCoordinator()
        val stopping = CompletableDeferred<Unit>()
        val release = CompletableDeferred<Unit>()
        val events = mutableListOf<String>()
        modes.exclusive {
            modes.registerWorker(Any()) {
                stopping.complete(Unit)
                release.await()
                events += "worker exited"
            }
        }
        val switch = launch(start = CoroutineStart.UNDISPATCHED) {
            modes.switchMode(SettingsDataStore.MODE_LOCAL,
                persistMode = { events += "mode published" },
                releaseModel = { events += "tail released"; Result.success(Unit) })
        }
        withTimeout(5_000) { stopping.await() }
        switch.cancel()
        assertFalse(switch.isCompleted)
        release.complete(Unit)
        withTimeout(5_000) { switch.join() }
        assertEquals(listOf("worker exited", "tail released", "mode published"), events)
    }

    @Test
    fun aDestroyedOldServiceCannotStopTheReplacementWorker() = runBlocking {
        val modes = InferenceModeCoordinator()
        val oldOwner = Any()
        val newOwner = Any()
        var oldStops = 0
        var newStops = 0
        modes.exclusive {
            modes.registerWorker(oldOwner) { oldStops++ }
            modes.stopWorkerLocked()
            modes.registerWorker(newOwner) { newStops++ }
        }
        modes.retireWorker(oldOwner)
        assertEquals(1, oldStops)
        assertEquals(0, newStops)
        modes.worker<Unit>({ SettingsDataStore.MODE_DISTRIBUTED }) { Result.success(Unit) }.getOrThrow()
        modes.retireWorker(newOwner)
        assertEquals(1, newStops)
    }

    @Test
    fun modeSwitchWaitsForAnAlreadyRunningLocalGeneration() = runBlocking {
        val modes = InferenceModeCoordinator()
        val generating = CompletableDeferred<Unit>()
        val finished = CompletableDeferred<Unit>()
        val events = mutableListOf<String>()
        val generation = async(start = CoroutineStart.UNDISPATCHED) {
            modes.local<Unit>({ SettingsDataStore.MODE_LOCAL }, { error("no worker") }) {
                generating.complete(Unit)
                withContext(NonCancellable) { finished.await() }
                events += "generation finished"
                Result.success(Unit)
            }
        }
        withTimeout(5_000) { generating.await() }
        val switch = async(start = CoroutineStart.UNDISPATCHED) {
            modes.switchMode(SettingsDataStore.MODE_DISTRIBUTED,
                persistMode = { events += "distributed published" },
                releaseModel = { events += "full released"; Result.success(Unit) })
        }
        try {
            assertFalse(switch.isCompleted)
            finished.complete(Unit)
            withTimeout(5_000) { generation.await(); switch.await() }
            assertEquals(listOf("generation finished", "full released", "distributed published"), events)
        } finally {
            finished.complete(Unit)
        }
    }
}
