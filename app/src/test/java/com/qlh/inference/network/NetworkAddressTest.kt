package com.qlh.inference.network

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * NetworkAddress 纯函数单元测试（JVM，无需设备）。
 *
 * 与 PC 端 tests/test_network_address.py 保持同一契约：
 * - 设置中保持裸 host（IPv6 不带方括号），仅在构建 URL 时加括号；
 * - IPv6 字面量必须用 [ ] 包裹，且 % 转义为 %25。
 */
class NetworkAddressTest {

    // ---- canonicalHost ----

    @Test
    fun `canonicalHost strips brackets and trims`() {
        assertEquals("fd7a:115c:a1e0::1", canonicalHost("[fd7a:115c:a1e0::1]"))
        assertEquals("fd7a:115c:a1e0::1", canonicalHost("  fd7a:115c:a1e0::1  "))
        assertEquals("192.168.1.20", canonicalHost("[192.168.1.20]"))
    }

    @Test
    fun `canonicalHost keeps bare hosts unchanged`() {
        assertEquals("localhost", canonicalHost("localhost"))
        assertEquals("100.90.1.2", canonicalHost("100.90.1.2"))
        assertEquals("node.ts.net", canonicalHost("node.ts.net"))
    }

    // ---- formatUrlHost ----

    @Test
    fun `formatUrlHost brackets ipv6 literals`() {
        assertEquals("[fd7a:115c:a1e0::10]", formatUrlHost("fd7a:115c:a1e0::10"))
        assertEquals("[::1]", formatUrlHost("::1"))
        // 已带括号的输入不重复包裹
        assertEquals("[fd7a:115c:a1e0::10]", formatUrlHost("[fd7a:115c:a1e0::10]"))
    }

    @Test
    fun `formatUrlHost escapes percent in ipv6 zone ids`() {
        assertEquals("[fe80::1%25eth0]", formatUrlHost("fe80::1%eth0"))
        assertEquals("[fe80::1%25wlan0]", formatUrlHost("[fe80::1%wlan0]"))
    }

    @Test
    fun `formatUrlHost leaves ipv4 and dns names bare`() {
        assertEquals("192.168.1.20", formatUrlHost("192.168.1.20"))
        assertEquals("localhost", formatUrlHost("localhost"))
        assertEquals("100.90.1.2", formatUrlHost("100.90.1.2"))
    }

    // ---- httpBaseUrl ----

    @Test
    fun `httpBaseUrl builds ipv6 url with brackets`() {
        assertEquals(
            "http://[fd7a:115c:a1e0::10]:8000",
            httpBaseUrl("fd7a:115c:a1e0::10", 8000),
        )
    }

    @Test
    fun `httpBaseUrl builds ipv4 and hostname urls`() {
        assertEquals("http://192.168.1.20:8000", httpBaseUrl("192.168.1.20", 8000))
        assertEquals("http://localhost:8080", httpBaseUrl("localhost", 8080))
        assertEquals("http://node.ts.net:443", httpBaseUrl("node.ts.net", 443))
    }

    @Test
    fun `httpBaseUrl handles already bracketed input`() {
        assertEquals(
            "http://[fd7a:115c:a1e0::10]:8000",
            httpBaseUrl("[fd7a:115c:a1e0::10]", 8000),
        )
    }
}
