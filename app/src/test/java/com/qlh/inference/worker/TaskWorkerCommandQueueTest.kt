package com.qlh.inference.worker

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TaskWorkerCommandQueueTest {
    @Test
    fun startStopAndResumeRunInArrivalOrder() = runBlocking {
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val firstStarted = CompletableDeferred<Unit>()
        val releaseFirst = CompletableDeferred<Unit>()
        val completed = CompletableDeferred<Unit>()
        val handled = mutableListOf<String>()
        val queue = TaskWorkerCommandQueue<String>(
            scope = scope,
            handle = { command ->
                if (command == "start") {
                    firstStarted.complete(Unit)
                    releaseFirst.await()
                }
                handled += command
                if (handled.size == 3) completed.complete(Unit)
            },
            onFailure = { _, failure -> throw failure },
        )
        try {
            assertTrue(queue.enqueue("start"))
            withTimeout(5_000) { firstStarted.await() }
            assertTrue(queue.enqueue("stop"))
            assertTrue(queue.enqueue("resume"))
            assertTrue(handled.isEmpty())
            releaseFirst.complete(Unit)
            withTimeout(5_000) { completed.await() }
            assertEquals(listOf("start", "stop", "resume"), handled)
        } finally {
            queue.close()
            scope.cancel()
        }
    }

    @Test
    fun aFailedCommandDoesNotDropFollowingStop() = runBlocking {
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val stopped = CompletableDeferred<Unit>()
        val failures = mutableListOf<String>()
        val queue = TaskWorkerCommandQueue<String>(
            scope = scope,
            handle = { command ->
                if (command == "broken-start") throw IllegalStateException("broken")
                if (command == "stop") stopped.complete(Unit)
            },
            onFailure = { command, _ -> failures += command },
        )
        try {
            assertTrue(queue.enqueue("broken-start"))
            assertTrue(queue.enqueue("stop"))
            withTimeout(5_000) { stopped.await() }
            assertEquals(listOf("broken-start"), failures)
            queue.close()
            assertFalse(queue.enqueue("resume"))
        } finally {
            queue.close()
            scope.cancel()
        }
    }
}
