package com.qlh.inference.control

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.qlh.inference.security.QuorumEd25519
import java.security.MessageDigest

internal class QuorumVoterException(val code: String) : IllegalArgumentException(code)

internal data class AndroidVoterIdentity(val voterId: String, val publicKey: ByteArray)

internal data class AndroidVoterSet(
    val clusterId: String,
    val epoch: Long,
    val voters: Map<String, AndroidVoterIdentity>,
) {
    val quorumSize: Int get() = voters.size / 2 + 1

    companion object {
        private val fields = setOf("schema_version", "document_type", "cluster_id", "voter_set_epoch", "voters")
        fun parse(value: JsonObject): AndroidVoterSet {
            requireFields(value, fields)
            require(value.string("schema_version") == SCHEMA && value.string("document_type") == "voter_set")
            val cluster = safeId(value.string("cluster_id"))
            val epoch = value.integer("voter_set_epoch", 0)
            val array = value["voters"]
            require(array != null && array.isJsonArray && array.asJsonArray.size() > 0)
            val identities = linkedMapOf<String, AndroidVoterIdentity>()
            array.asJsonArray.forEach { element ->
                require(element.isJsonObject)
                val voter = element.asJsonObject
                requireFields(voter, setOf("voter_id", "public_key"))
                val id = safeId(voter.string("voter_id"))
                val key = QuorumEd25519.decode(voter.string("public_key"), 32)
                require(identities.put(id, AndroidVoterIdentity(id, key)) == null)
            }
            return AndroidVoterSet(cluster, epoch, identities.toSortedMap())
        }
    }
}

internal data class AndroidQuorumCertificate(
    val json: JsonObject,
    val clusterId: String,
    val epoch: Long,
    val term: Long,
    val leaderId: String,
    val leaseId: String,
    val expiresAtMs: Long,
    val watermarks: Map<String, Pair<Long, String>>,
    val signatures: List<AndroidVoterSignature>,
) {
    fun signingPayload(): JsonObject = JsonObject().apply {
        addProperty("schema_version", SCHEMA)
        addProperty("document_type", "quorum_certificate")
        addProperty("cluster_id", clusterId)
        addProperty("voter_set_epoch", epoch)
        addProperty("term", term)
        addProperty("leader_id", leaderId)
        addProperty("lease_id", leaseId)
        addProperty("expires_at_ms", expiresAtMs)
        if (watermarks.isNotEmpty()) {
            val array = JsonArray()
            watermarks.toSortedMap().forEach { (workflow, point) ->
                array.add(JsonObject().apply {
                    addProperty("workflow_id", workflow)
                    addProperty("last_sequence", point.first)
                    addProperty("checkpoint_digest", point.second)
                })
            }
            add("commit_watermark", array)
        }
    }

    fun signingBytes(): ByteArray = CanonicalJson.encode(signingPayload()).toByteArray(Charsets.UTF_8)

    fun toJson(): JsonObject = signingPayload().apply {
        val array = JsonArray()
        signatures.sortedBy { it.voterId }.forEach { signature ->
            array.add(JsonObject().apply {
                addProperty("voter_id", signature.voterId)
                addProperty("algorithm", "ed25519")
                addProperty("signature", signature.encoded)
            })
        }
        add("signatures", array)
    }

    fun digest(): String = MessageDigest.getInstance("SHA-256")
        .digest(CanonicalJson.encode(toJson()).toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    fun withSignature(signature: AndroidVoterSignature): AndroidQuorumCertificate = copy(
        signatures = (signatures.filterNot { it.voterId == signature.voterId } + signature)
            .sortedBy { it.voterId },
    )

    companion object {
        private val required = setOf(
            "schema_version", "document_type", "cluster_id", "voter_set_epoch", "term",
            "leader_id", "lease_id", "expires_at_ms", "signatures",
        )
        fun parse(value: JsonObject): AndroidQuorumCertificate {
            require(value.keySet().containsAll(required) &&
                value.keySet().all { it in required || it == "commit_watermark" })
            require(value.string("schema_version") == SCHEMA &&
                value.string("document_type") == "quorum_certificate")
            val signaturesElement = value["signatures"]
            require(signaturesElement != null && signaturesElement.isJsonArray)
            val signatures = signaturesElement.asJsonArray.map { element ->
                require(element.isJsonObject)
                val signature = element.asJsonObject
                requireFields(signature, setOf("voter_id", "algorithm", "signature"))
                require(signature.string("algorithm") == "ed25519")
                val encoded = signature.string("signature")
                QuorumEd25519.decode(encoded, 64)
                AndroidVoterSignature(safeId(signature.string("voter_id")), encoded)
            }
            val watermarks = linkedMapOf<String, Pair<Long, String>>()
            val watermarkElement = value.get("commit_watermark")
            if (watermarkElement != null) {
                require(watermarkElement.isJsonArray)
                watermarkElement.asJsonArray.forEach { element ->
                    require(element.isJsonObject)
                    val watermark = element.asJsonObject
                    requireFields(watermark, setOf("workflow_id", "last_sequence", "checkpoint_digest"))
                    val workflow = safeId(watermark.string("workflow_id"))
                    val sequence = watermark.integer("last_sequence", 0)
                    val digest = watermark.string("checkpoint_digest")
                    require(SHA256.matches(digest) && watermarks.put(workflow, sequence to digest) == null)
                }
            }
            return AndroidQuorumCertificate(
                json = value.deepCopy(),
                clusterId = safeId(value.string("cluster_id")),
                epoch = value.integer("voter_set_epoch", 0),
                term = value.integer("term", 1),
                leaderId = safeId(value.string("leader_id")),
                leaseId = safeId(value.string("lease_id")),
                expiresAtMs = value.integer("expires_at_ms", 1),
                watermarks = watermarks.toSortedMap(),
                signatures = signatures,
            )
        }
    }
}

internal data class AndroidVoterSignature(val voterId: String, val encoded: String)

internal fun validateAndroidCertificate(
    certificate: AndroidQuorumCertificate,
    voterSet: AndroidVoterSet,
    nowMs: Long,
) {
    if (certificate.clusterId != voterSet.clusterId || certificate.epoch != voterSet.epoch ||
        certificate.leaderId !in voterSet.voters || certificate.expiresAtMs <= nowMs) {
        throw QuorumVoterException("quorum_vote_rejected")
    }
    val seen = mutableSetOf<String>()
    val bytes = certificate.signingBytes()
    certificate.signatures.forEach { signature ->
        val identity = voterSet.voters[signature.voterId]
            ?: throw QuorumVoterException("quorum_vote_rejected")
        if (!seen.add(signature.voterId) || !QuorumEd25519.verify(
                identity.publicKey, bytes, QuorumEd25519.decode(signature.encoded, 64))) {
            throw QuorumVoterException("quorum_vote_rejected")
        }
    }
    if (seen.size < voterSet.quorumSize) throw QuorumVoterException("quorum_vote_rejected")
}

internal object CanonicalJson {
    fun encode(value: JsonElement): String = when {
        value.isJsonNull -> "null"
        value.isJsonObject -> value.asJsonObject.entrySet().sortedBy { it.key }.joinToString(",", "{", "}") {
            quote(it.key) + ":" + encode(it.value)
        }
        value.isJsonArray -> value.asJsonArray.joinToString(",", "[", "]") { encode(it) }
        value.asJsonPrimitive.isString -> quote(value.asString)
        value.asJsonPrimitive.isBoolean -> value.asBoolean.toString()
        else -> value.asJsonPrimitive.asString
    }

    private fun quote(value: String): String = buildString {
        append('"')
        value.forEach { character ->
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\b' -> append("\\b")
                '\u000c' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (character.code < 0x20 || character.code > 0x7e) {
                    append("\\u%04x".format(character.code))
                } else append(character)
            }
        }
        append('"')
    }
}

internal fun safeId(value: String): String {
    require(ID.matches(value))
    return value
}

private fun requireFields(value: JsonObject, fields: Set<String>) {
    require(value.keySet() == fields)
}

private fun JsonObject.string(key: String): String {
    val value = get(key)
    require(value != null && value.isJsonPrimitive && value.asJsonPrimitive.isString)
    return value.asString
}

private fun JsonObject.integer(key: String, minimum: Long): Long {
    val value = get(key)
    require(value != null && value.isJsonPrimitive && value.asJsonPrimitive.isNumber &&
        INTEGER.matches(value.asString))
    return value.asString.toLong().also { require(it >= minimum) }
}

private const val SCHEMA = "qlh.cluster.control.v1"
private val ID = Regex("[A-Za-z0-9_.:/-]{1,256}")
private val INTEGER = Regex("0|[1-9][0-9]*")
private val SHA256 = Regex("[0-9a-f]{64}")
