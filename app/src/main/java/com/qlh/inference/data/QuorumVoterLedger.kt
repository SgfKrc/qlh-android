package com.qlh.inference.data

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.google.gson.JsonObject
import com.qlh.inference.control.AndroidQuorumCertificate
import com.qlh.inference.control.AndroidVoterSignature
import com.qlh.inference.control.AndroidVoterSet
import com.qlh.inference.control.QuorumVoterException
import com.qlh.inference.control.safeId
import com.qlh.inference.control.validateAndroidCertificate
import com.qlh.inference.security.QuorumEd25519
import java.security.MessageDigest

private fun quorumLedgerDatabaseName(voterId: String): String {
    val digest = MessageDigest.getInstance("SHA-256").digest(voterId.toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
    return "quorum-$digest.sqlite3"
}

internal class QuorumVoterLedger(
    context: Context,
    private val voterId: String,
    private val voterSet: AndroidVoterSet,
    databaseName: String = quorumLedgerDatabaseName(voterId),
) : SQLiteOpenHelper(context.applicationContext, databaseName, null, DATABASE_VERSION) {
    private val gate = Any()

    init {
        require(voterId in voterSet.voters)
        setWriteAheadLoggingEnabled(true)
        writableDatabase
    }

    override fun onConfigure(db: SQLiteDatabase) {
        db.setForeignKeyConstraintsEnabled(true)
        db.rawQuery("PRAGMA busy_timeout=5000", null).use { it.moveToFirst() }
        db.rawQuery("PRAGMA synchronous=FULL", null).use { it.moveToFirst() }
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""CREATE TABLE quorum_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
        db.execSQL("""CREATE TABLE quorum_term_state (
            id INTEGER PRIMARY KEY CHECK (id=1), promised_term INTEGER NOT NULL CHECK (promised_term>=0),
            promised_leader TEXT NOT NULL, active_term INTEGER NOT NULL CHECK (active_term>=0),
            active_leader TEXT NOT NULL, active_lease_id TEXT NOT NULL, active_expires_at_ms INTEGER,
            active_certificate_digest TEXT NOT NULL)""")
        db.execSQL("""CREATE TABLE quorum_recovery_floors (
            domain_id TEXT PRIMARY KEY, last_sequence INTEGER NOT NULL CHECK (last_sequence>=0),
            checkpoint_digest TEXT NOT NULL, floor_state TEXT NOT NULL
            CHECK (floor_state IN ('accepted','committed')))""")
        db.execSQL("""CREATE TABLE quorum_votes (
            voter_set_epoch INTEGER NOT NULL, term INTEGER NOT NULL CHECK (term>0), voter_id TEXT NOT NULL,
            leader_id TEXT NOT NULL, lease_id TEXT NOT NULL, expires_at_ms INTEGER NOT NULL,
            signature TEXT NOT NULL, PRIMARY KEY (voter_set_epoch, term, voter_id))""")
        db.execSQL("""INSERT INTO quorum_term_state
            (id,promised_term,promised_leader,active_term,active_leader,active_lease_id,
             active_expires_at_ms,active_certificate_digest)
            VALUES (1,0,'',0,'','',NULL,'')""")
        val metadata = mapOf(
            "schema_version" to "1",
            "cluster_id" to voterSet.clusterId,
            "voter_id" to voterId,
            "voter_set_epoch" to voterSet.epoch.toString(),
            "recovery_floor_schema" to "2",
        )
        metadata.forEach { (key, value) ->
            db.insertOrThrow("quorum_metadata", null, ContentValues().apply {
                put("key", key)
                put("value", value)
            })
        }
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        throw QuorumVoterException("quorum_ledger_corrupt")
    }

    override fun onOpen(db: SQLiteDatabase) {
        super.onOpen(db)
        val expected = mapOf(
            "schema_version" to "1",
            "cluster_id" to voterSet.clusterId,
            "voter_id" to voterId,
            "voter_set_epoch" to voterSet.epoch.toString(),
        )
        expected.forEach { (key, value) ->
            val actual = db.rawQuery("SELECT value FROM quorum_metadata WHERE key=?", arrayOf(key)).use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
            if (actual != value) throw QuorumVoterException("quorum_ledger_corrupt")
        }
        db.rawQuery("PRAGMA synchronous=FULL", null).use { it.moveToFirst() }
    }

    fun reserveTerm(leader: String, nowMs: Long): Long = transaction { db ->
        requireLeader(leader)
        val state = state(db)
        val activeExpiry = state.nullableLong("active_expires_at_ms")
        if (activeExpiry != null && activeExpiry > nowMs) throw QuorumVoterException("quorum_active_lease")
        requireAcceptedLease(db, leader, -1L, null, nowMs)
        val term = maxOf(state.getLong("promised_term"), state.getLong("active_term")) + 1
        db.update("quorum_term_state", ContentValues().apply {
            put("promised_term", term)
            put("promised_leader", leader)
        }, "id=1", null)
        term
    }

    fun prepare(leader: String, term: Long, nowMs: Long) = transaction { db ->
        requireLeader(leader)
        if (term <= 0) throw QuorumVoterException("quorum_term_stale")
        val state = state(db)
        requireAcceptedLease(db, leader, term, null, nowMs)
        val activeExpiry = state.nullableLong("active_expires_at_ms")
        val activeTerm = state.getLong("active_term")
        if (activeExpiry != null && activeExpiry > nowMs && term > activeTerm) {
            throw QuorumVoterException("quorum_active_lease")
        }
        val promised = state.getLong("promised_term")
        val promisedLeader = state.getString("promised_leader")
        if (term < promised) throw QuorumVoterException("quorum_term_stale")
        if (term == promised && promisedLeader.isNotEmpty() && promisedLeader != leader) {
            throw QuorumVoterException("quorum_term_conflict")
        }
        if (term > promised || promisedLeader != leader) {
            db.update("quorum_term_state", ContentValues().apply {
                put("promised_term", term)
                put("promised_leader", leader)
            }, "id=1", null)
        }
        Unit
    }

    fun signVote(certificate: AndroidQuorumCertificate, seed: ByteArray, nowMs: Long): AndroidVoterSignature = transaction { db ->
        if (certificate.epoch != voterSet.epoch || certificate.clusterId != voterSet.clusterId ||
            certificate.leaderId !in voterSet.voters || certificate.expiresAtMs <= nowMs) {
            throw QuorumVoterException("quorum_vote_rejected")
        }
        requirePreservedFloors(db, certificate)
        if (certificate.watermarks.isNotEmpty()) {
            throw QuorumVoterException("quorum_watermark_unavailable")
        }
        requireAcceptedLease(db, certificate.leaderId, certificate.term, certificate.leaseId, nowMs)
        val state = state(db)
        if (state.getLong("promised_term") != certificate.term ||
            state.getString("promised_leader") != certificate.leaderId) {
            throw QuorumVoterException("quorum_vote_rejected")
        }
        val activeExpiry = state.nullableLong("active_expires_at_ms")
        if (activeExpiry != null && activeExpiry > nowMs &&
            (state.getLong("active_term") != certificate.term ||
                state.getString("active_leader") != certificate.leaderId ||
                state.getString("active_lease_id") != certificate.leaseId)) {
            throw QuorumVoterException("quorum_active_lease")
        }
        val existing = db.rawQuery(
            "SELECT leader_id,lease_id,expires_at_ms FROM quorum_votes WHERE voter_set_epoch=? AND term=? AND voter_id=?",
            arrayOf(voterSet.epoch.toString(), certificate.term.toString(), voterId),
        ).use { cursor -> if (cursor.moveToFirst()) cursor.row() else null }
        if (existing != null) {
            if (existing.getString("leader_id") != certificate.leaderId ||
                existing.getString("lease_id") != certificate.leaseId) {
                throw QuorumVoterException("quorum_duplicate_vote")
            }
            if (existing.getLong("expires_at_ms") > certificate.expiresAtMs) {
                throw QuorumVoterException("quorum_expiry_regressed")
            }
        }
        val encoded = QuorumEd25519.encode(QuorumEd25519.sign(seed, certificate.signingBytes()))
        val values = ContentValues().apply {
            put("voter_set_epoch", voterSet.epoch)
            put("term", certificate.term)
            put("voter_id", voterId)
            put("leader_id", certificate.leaderId)
            put("lease_id", certificate.leaseId)
            put("expires_at_ms", certificate.expiresAtMs)
            put("signature", encoded)
        }
        db.insertWithOnConflict("quorum_votes", null, values, SQLiteDatabase.CONFLICT_REPLACE)
        saveFloors(db, certificate, "accepted")
        AndroidVoterSignature(voterId, encoded)
    }

    fun commitCertificate(certificate: AndroidQuorumCertificate, nowMs: Long) = transaction { db ->
        validateAndroidCertificate(certificate, voterSet, nowMs)
        if (voterId !in certificate.signatures.map { it.voterId }) {
            throw QuorumVoterException("quorum_vote_rejected")
        }
        requirePreservedFloors(db, certificate)
        requireAcceptedLease(db, certificate.leaderId, certificate.term, certificate.leaseId, nowMs)
        val localVote = db.rawQuery(
            "SELECT leader_id,lease_id,expires_at_ms,signature FROM quorum_votes WHERE voter_set_epoch=? AND term=? AND voter_id=?",
            arrayOf(certificate.epoch.toString(), certificate.term.toString(), voterId),
        ).use { cursor -> if (cursor.moveToFirst()) cursor.row() else null }
        val ownSignature = certificate.signatures.single { it.voterId == voterId }
        if (localVote == null || localVote.getString("leader_id") != certificate.leaderId ||
            localVote.getString("lease_id") != certificate.leaseId ||
            localVote.getLong("expires_at_ms") != certificate.expiresAtMs ||
            localVote.getString("signature") != ownSignature.encoded) {
            throw QuorumVoterException("quorum_vote_rejected")
        }
        val state = state(db)
        val activeExpiry = state.nullableLong("active_expires_at_ms")
        if (activeExpiry != null && activeExpiry > nowMs &&
            (state.getLong("active_term") != certificate.term ||
                state.getString("active_leader") != certificate.leaderId ||
                state.getString("active_lease_id") != certificate.leaseId)) {
            throw QuorumVoterException("quorum_active_lease")
        }
        if (certificate.term < state.getLong("active_term")) throw QuorumVoterException("quorum_term_stale")
        db.update("quorum_term_state", ContentValues().apply {
            put("active_term", certificate.term)
            put("active_leader", certificate.leaderId)
            put("active_lease_id", certificate.leaseId)
            put("active_expires_at_ms", certificate.expiresAtMs)
            put("active_certificate_digest", certificate.digest())
        }, "id=1", null)
        saveFloors(db, certificate, "committed")
        Unit
    }

    fun snapshot(): JsonObject = synchronized(gate) {
        state(readableDatabase).let { row -> JsonObject().apply {
            addProperty("voter_id", voterId)
            addProperty("cluster_id", voterSet.clusterId)
            addProperty("voter_set_epoch", voterSet.epoch)
            addProperty("promised_term", row.getLong("promised_term"))
            addProperty("promised_leader", row.getString("promised_leader"))
            addProperty("active_term", row.getLong("active_term"))
            addProperty("active_leader", row.getString("active_leader"))
            addProperty("active_lease_id", row.getString("active_lease_id"))
            val expiry = row.nullableLong("active_expires_at_ms")
            if (expiry == null) add("active_expires_at_ms", com.google.gson.JsonNull.INSTANCE)
            else addProperty("active_expires_at_ms", expiry)
            addProperty("active_certificate_digest", row.getString("active_certificate_digest"))
        } }
    }

    private fun requireLeader(leader: String) {
        runCatching { safeId(leader) }.getOrElse {
            throw QuorumVoterException("quorum_invalid_identity")
        }
        if (leader.isBlank()) throw QuorumVoterException("quorum_invalid_identity")
        if (leader !in voterSet.voters) throw QuorumVoterException("quorum_invalid_leader")
    }

    private fun requireAcceptedLease(db: SQLiteDatabase, leader: String, term: Long, lease: String?, now: Long) {
        db.rawQuery("SELECT term,leader_id,lease_id FROM quorum_votes WHERE expires_at_ms>?", arrayOf(now.toString()))
            .use { cursor ->
                while (cursor.moveToNext()) {
                    val leaseMatches = lease == null || cursor.getString(2) == lease
                    if (cursor.getLong(0) != term || cursor.getString(1) != leader || !leaseMatches) {
                        throw QuorumVoterException("quorum_active_lease")
                    }
                }
            }
    }

    private fun requirePreservedFloors(db: SQLiteDatabase, certificate: AndroidQuorumCertificate) {
        val marker = db.rawQuery(
            "SELECT value FROM quorum_metadata WHERE key='recovery_floor_schema'", null,
        ).use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
        if (marker != "2") throw QuorumVoterException("quorum_watermark_unavailable")
        db.rawQuery("SELECT domain_id,last_sequence,checkpoint_digest FROM quorum_recovery_floors", null)
            .use { cursor ->
                while (cursor.moveToNext()) {
                    val proposed = certificate.watermarks[cursor.getString(0)]
                    if (proposed == null || proposed.first < cursor.getLong(1) ||
                        (proposed.first == cursor.getLong(1) && proposed.second != cursor.getString(2))) {
                        throw QuorumVoterException("quorum_watermark_unheld")
                    }
                }
            }
    }

    private fun saveFloors(db: SQLiteDatabase, certificate: AndroidQuorumCertificate, state: String) {
        certificate.watermarks.forEach { (domain, point) ->
            val previous = db.rawQuery(
                "SELECT last_sequence,checkpoint_digest,floor_state FROM quorum_recovery_floors WHERE domain_id=?",
                arrayOf(domain),
            ).use { cursor -> if (cursor.moveToFirst()) cursor.row() else null }
            if (previous != null && point.first < previous.getLong("last_sequence")) return@forEach
            if (previous != null && point.first == previous.getLong("last_sequence") &&
                point.second != previous.getString("checkpoint_digest")) {
                throw QuorumVoterException("quorum_watermark_unheld")
            }
            val floorState = if (previous != null && previous.getString("floor_state") == "committed" &&
                previous.getLong("last_sequence") == point.first) "committed" else state
            db.insertWithOnConflict("quorum_recovery_floors", null, ContentValues().apply {
                put("domain_id", domain)
                put("last_sequence", point.first)
                put("checkpoint_digest", point.second)
                put("floor_state", floorState)
            }, SQLiteDatabase.CONFLICT_REPLACE)
        }
    }

    private fun state(db: SQLiteDatabase): ContentValues = db.rawQuery(
        "SELECT promised_term,promised_leader,active_term,active_leader,active_lease_id,"
            + "active_expires_at_ms,active_certificate_digest FROM quorum_term_state WHERE id=1", null,
    ).use { cursor ->
        if (!cursor.moveToFirst()) throw QuorumVoterException("quorum_ledger_corrupt")
        cursor.row()
    }

    private fun <T> transaction(block: (SQLiteDatabase) -> T): T = synchronized(gate) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            block(db).also { db.setTransactionSuccessful() }
        } finally {
            db.endTransaction()
        }
    }

    private fun Cursor.row(): android.content.ContentValues = ContentValues().also { values ->
        for (index in 0 until columnCount) {
            if (isNull(index)) values.putNull(getColumnName(index)) else values.put(getColumnName(index), getString(index))
        }
    }

    private fun android.content.ContentValues.getLong(key: String): Long = getAsString(key).toLong()
    private fun android.content.ContentValues.getString(key: String): String = getAsString(key)
    private fun android.content.ContentValues.nullableLong(key: String): Long? = if (containsKey(key) && get(key) != null) getAsString(key).toLong() else null

    companion object {
        private const val DATABASE_VERSION = 1
    }
}

internal class AndroidQuorumVoter(
    private val voterId: String,
    private val voterSet: AndroidVoterSet,
    seed: ByteArray,
    private val ledger: QuorumVoterLedger,
) : AutoCloseable {
    private val seed = seed.copyOf()
    @Volatile private var closed = false

    @Synchronized
    fun handle(payload: JsonObject, nowMs: Long): JsonObject = try {
        if (closed) throw QuorumVoterException("quorum_unavailable")
        when (payload.get("op")?.takeIf { it.isJsonPrimitive }?.asString.orEmpty()) {
            "reserve_term" -> JsonObject().apply {
                addProperty("ok", true)
                addProperty("term", ledger.reserveTerm(payload.requiredString("leader_id"), nowMs))
            }
            "prepare" -> {
                ledger.prepare(payload.requiredString("leader_id"), payload.requiredLong("term"), nowMs)
                JsonObject().apply { addProperty("ok", true) }
            }
            "sign_vote" -> {
                val element = payload.get("certificate")
                require(element != null && element.isJsonObject)
                val signature = ledger.signVote(AndroidQuorumCertificate.parse(element.asJsonObject), seed, nowMs)
                JsonObject().apply {
                    addProperty("ok", true)
                    add("signature", JsonObject().apply {
                        addProperty("voter_id", signature.voterId)
                        addProperty("algorithm", "ed25519")
                        addProperty("signature", signature.encoded)
                    })
                }
            }
            "commit_certificate" -> {
                val element = payload.get("certificate")
                require(element != null && element.isJsonObject)
                ledger.commitCertificate(AndroidQuorumCertificate.parse(element.asJsonObject), nowMs)
                JsonObject().apply { addProperty("ok", true) }
            }
            "snapshot" -> JsonObject().apply { addProperty("ok", true); add("snapshot", ledger.snapshot()) }
            "leader_certificate" -> JsonObject().apply {
                addProperty("ok", true)
                add("certificate", com.google.gson.JsonNull.INSTANCE)
                addProperty("tcp_port", 0)
            }
            else -> throw QuorumVoterException("quorum_unavailable")
        }
    } catch (error: QuorumVoterException) {
        JsonObject().apply { addProperty("ok", false); addProperty("code", error.code) }
    } catch (_: Exception) {
        JsonObject().apply { addProperty("ok", false); addProperty("code", "quorum_unavailable") }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        seed.fill(0)
    }

    private fun JsonObject.requiredString(key: String): String {
        val value = get(key)
        require(value != null && value.isJsonPrimitive && value.asJsonPrimitive.isString)
        return value.asString
    }

    private fun JsonObject.requiredLong(key: String): Long {
        val value = get(key)
        require(value != null && value.isJsonPrimitive && value.asJsonPrimitive.isNumber &&
            value.asString.matches(Regex("0|[1-9][0-9]*")))
        return value.asString.toLong()
    }
}
