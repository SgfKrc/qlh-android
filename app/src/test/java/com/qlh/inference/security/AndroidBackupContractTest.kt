package com.qlh.inference.security

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class AndroidBackupContractTest {
    private val exclusions = listOf(
        "<exclude domain=\"sharedpref\" path=\"qlh_auth_session.xml\" />",
        "<exclude domain=\"sharedpref\" path=\"qlh_cluster_credential.xml\" />",
        "<exclude domain=\"file\" path=\"datastore/\" />",
        "<exclude domain=\"file\" path=\"logs/\" />",
        "<exclude domain=\"database\" path=\".\" />",
    )

    @Test
    fun `manifest wires both legacy and modern backup policies`() {
        val manifest = sourceFile("src/main/AndroidManifest.xml").readText()

        assertTrue(manifest.contains("android:fullBackupContent=\"@xml/backup_rules\""))
        assertTrue(manifest.contains("android:dataExtractionRules=\"@xml/data_extraction_rules\""))
    }

    @Test
    fun `backup policies exclude credentials datastore logs and databases`() {
        val legacyRules = sourceFile("src/main/res/xml/backup_rules.xml").readText()
        val modernRules = sourceFile("src/main/res/xml/data_extraction_rules.xml").readText()

        exclusions.forEach { exclusion ->
            assertTrue("legacy backup must exclude $exclusion", legacyRules.contains(exclusion))
            assertEquals(
                "cloud backup and device transfer must both exclude $exclusion",
                2,
                modernRules.windowed(exclusion.length).count { it == exclusion },
            )
        }
    }

    private fun sourceFile(moduleRelativePath: String): File {
        var current = File(
            System.getProperty("user.dir") ?: error("user.dir is unavailable"),
        ).canonicalFile
        repeat(8) {
            listOf(
                File(current, moduleRelativePath),
                File(current, "app/$moduleRelativePath"),
                File(current, "android/app/$moduleRelativePath"),
            ).firstOrNull(File::isFile)?.let { return it }
            current = current.parentFile ?: return@repeat
        }
        error("unable to locate Android source file: $moduleRelativePath")
    }
}
