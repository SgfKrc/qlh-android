package com.qlh.inference.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.junit.Assert.assertEquals
import org.junit.Test

class ThemeTokenTest {
    @Test
    fun neutralSchemesKeepBlackAndWhiteAsThePrimaryReadingContrast() {
        assertEquals(Color(0xFFFFFFFF), BackgroundLight)
        assertEquals(Color(0xFF151515), OnBackgroundLight)
        assertEquals(Color(0xFF000000), BackgroundDark)
        assertEquals(Color(0xFFF5F5F5), OnBackgroundDark)
        assertEquals(Color(0xFF151515), PrimaryLight)
        assertEquals(Color(0xFFF3F3F3), PrimaryDark)
    }

    @Test
    fun sharedTokensKeepCompactCornersAndZeroLetterSpacing() {
        assertEquals(8.dp, QlhShapeTokens.control)
        assertEquals(12.dp, QlhShapeTokens.dialog)
        assertEquals(16.dp, QlhUiTokens.pageHorizontal)
        assertEquals(1.dp, QlhUiTokens.hairline)
        assertEquals(48.dp, QlhUiTokens.touchTarget)
        assertEquals(0.sp, QlhTypography.titleMedium.letterSpacing)
        assertEquals(0.sp, QlhTypography.bodyMedium.letterSpacing)
        assertEquals(0.sp, QlhTypography.labelMedium.letterSpacing)
    }

    @Test
    fun brandAccentIsSeparateFromBlackAndWhiteReadingBase() {
        assertEquals(Color(0xFFD8B4FF), BrandAccentDark)
        assertEquals(Color(0xFF624477), BrandAccentLight)
        assertEquals(Color(0xFF000000), BackgroundDark)
        assertEquals(Color(0xFFFFFFFF), BackgroundLight)
    }

    @Test
    fun goldAndNeonGreenStaySemanticAccents() {
        assertEquals(Color(0xFFD9C27A), BrandGoldDark)
        assertEquals(Color(0xFFC7FF3D), NeonGreenDark)
        assertEquals(Color(0xFF354500), NeonGreenContainerDark)
    }
}
