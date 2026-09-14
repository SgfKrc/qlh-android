package com.qlh.inference.update

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidUpdateContractTest {
    private fun manifest(tag: String, assets: String): String =
        """{"schema_version":1,"tag":"$tag","assets":[$assets]}"""

    private fun asset(
        variant: String,
        name: String = "QLH-Inference-$variant-v0.1.8.3.apk",
        sha256: String = "a".repeat(64),
        url: String = "/$name",
    ): String = """{
        "name":"$name","url":"$url","size":12345,"sha256":"$sha256",
        "platform":"android","variant":"$variant","arch":"any","kind":"installer"
    }"""

    @Test
    fun `selects only newer apk for current flavour`() {
        val selected = selectAndroidUpdate(
            manifest("0.1.8.3", "${asset("lite")},${asset("full")}"),
            currentVersion = "0.1.8.2",
            variant = "full",
        ).getOrThrow()

        assertEquals("QLH-Inference-full-v0.1.8.3.apk", selected?.name)
        assertEquals("0.1.8.3", selected?.version)
    }

    @Test
    fun `does not downgrade or reinstall same release`() {
        assertNull(
            selectAndroidUpdate(
                manifest("0.1.8.2", asset("full")),
                currentVersion = "0.1.8.2-debug",
                variant = "full",
            ).getOrThrow(),
        )
        assertNull(
            selectAndroidUpdate(
                manifest("0.1.8.1", asset("full")),
                currentVersion = "0.1.8.2",
                variant = "full",
            ).getOrThrow(),
        )
    }

    @Test
    fun `rejects invalid digest duplicate target and aab`() {
        assertTrue(
            selectAndroidUpdate(
                manifest("0.1.8.3", asset("full", sha256 = "bad")),
                "0.1.8.2",
                "full",
            ).isFailure,
        )
        assertTrue(
            selectAndroidUpdate(
                manifest("0.1.8.3", "${asset("full")},${asset("full", name = "other.apk")}"),
                "0.1.8.2",
                "full",
            ).isFailure,
        )
        assertTrue(
            selectAndroidUpdate(
                manifest("0.1.8.3", asset("full", name = "bundle.aab")),
                "0.1.8.2",
                "full",
            ).isFailure,
        )
    }

    @Test
    fun `builds ipv6 source and resolves local relative asset`() {
        assertEquals(
            "http://[fd7a:115c:a1e0::10]:9090/latest.json",
            defaultAndroidUpdateSources("fd7a:115c:a1e0::10").first(),
        )
        assertEquals(
            "http://[fd7a:115c:a1e0::10]:9090/QLH.apk",
            resolveUpdateAssetUrl(
                "http://[fd7a:115c:a1e0::10]:9090/latest.json",
                "/QLH.apk",
            ),
        )
    }
}
