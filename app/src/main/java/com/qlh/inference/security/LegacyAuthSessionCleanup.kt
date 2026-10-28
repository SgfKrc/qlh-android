package com.qlh.inference.security

import android.content.Context
import java.security.KeyStore

internal object LegacyAuthSessionCleanup {
    private const val STORE_NAME = "qlh_auth_session"
    private const val KEY_ALIAS = "com.qlh.inference.auth.session.v1"

    fun clear(context: Context) {
        runCatching { context.deleteSharedPreferences(STORE_NAME) }
        runCatching {
            KeyStore.getInstance("AndroidKeyStore").apply { load(null) }.deleteEntry(KEY_ALIAS)
        }
    }
}
