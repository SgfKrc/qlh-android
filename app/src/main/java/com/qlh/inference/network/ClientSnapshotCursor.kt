package com.qlh.inference.network

/** Per owner stream cursor; reboot requires a newer observation before revisions may restart. */
internal class ClientSnapshotCursor {
    private var previous: Triple<String, Long, Double>? = null
    @Synchronized fun accept(version: Int, revision: Long?, identity: Map<String, String>, observed: Double, validity: String): Boolean {
        val boot = identity["boot_id"]
        if (version != 2 || revision == null || revision < 0 || boot.isNullOrBlank() ||
            !observed.isFinite() || observed <= 0 || validity != "valid") return false
        val prior = previous
        if (prior != null && (observed < prior.third || (boot == prior.first && revision < prior.second))) return false
        previous = Triple(boot, revision, observed)
        return true
    }
}
