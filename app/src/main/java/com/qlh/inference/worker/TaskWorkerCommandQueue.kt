package com.qlh.inference.worker

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

internal class TaskWorkerCommandQueue<T>(
    scope: CoroutineScope,
    private val handle: suspend (T) -> Unit,
    private val onFailure: suspend (T, Exception) -> Unit,
) {
    private val commands = Channel<T>(Channel.UNLIMITED)

    init {
        scope.launch {
            for (command in commands) {
                try {
                    handle(command)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    onFailure(command, error)
                }
            }
        }
    }

    fun enqueue(command: T): Boolean = commands.trySend(command).isSuccess

    fun close() {
        commands.cancel()
    }
}
