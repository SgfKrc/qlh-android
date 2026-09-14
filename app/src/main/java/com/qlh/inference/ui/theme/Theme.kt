package com.qlh.inference.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = PrimaryLight,
    onPrimary = OnPrimaryLight,
    primaryContainer = PrimaryContainerLight,
    onPrimaryContainer = OnPrimaryContainerLight,
    secondary = SecondaryLight,
    onSecondary = OnSecondaryLight,
    secondaryContainer = SecondaryContainerLight,
    onSecondaryContainer = OnSecondaryContainerLight,
    tertiary = BrandAccentLight,
    onTertiary = OnBrandAccentLight,
    tertiaryContainer = BrandAccentContainerLight,
    onTertiaryContainer = OnBrandAccentContainerLight,
    error = ErrorLight,
    onError = OnErrorLight,
    errorContainer = ErrorContainerLight,
    onErrorContainer = OnErrorContainerLight,
    background = BackgroundLight,
    onBackground = OnBackgroundLight,
    surface = SurfaceLight,
    onSurface = OnSurfaceLight,
    surfaceVariant = SurfaceVariantLight,
    onSurfaceVariant = OnSurfaceVariantLight,
    outline = OutlineLight,
    outlineVariant = OutlineVariantLight,
    inverseSurface = InverseSurfaceLight,
    inverseOnSurface = InverseOnSurfaceLight,
    inversePrimary = InversePrimaryLight,
    surfaceDim = SurfaceDimLight,
    surfaceBright = SurfaceBrightLight,
    surfaceContainerLowest = SurfaceContainerLowestLight,
    surfaceContainerLow = SurfaceContainerLowLight,
    surfaceContainer = SurfaceContainerLight,
    surfaceContainerHigh = SurfaceContainerHighLight,
    surfaceContainerHighest = SurfaceContainerHighestLight
)

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryDark,
    onPrimary = OnPrimaryDark,
    primaryContainer = PrimaryContainerDark,
    onPrimaryContainer = OnPrimaryContainerDark,
    secondary = SecondaryDark,
    onSecondary = OnSecondaryDark,
    secondaryContainer = SecondaryContainerDark,
    onSecondaryContainer = OnSecondaryContainerDark,
    tertiary = BrandAccentDark,
    onTertiary = OnBrandAccentDark,
    tertiaryContainer = BrandAccentContainerDark,
    onTertiaryContainer = OnBrandAccentContainerDark,
    error = ErrorDark,
    onError = OnErrorDark,
    errorContainer = ErrorContainerDark,
    onErrorContainer = OnErrorContainerDark,
    background = BackgroundDark,
    onBackground = OnBackgroundDark,
    surface = SurfaceDark,
    onSurface = OnSurfaceDark,
    surfaceVariant = SurfaceVariantDark,
    onSurfaceVariant = OnSurfaceVariantDark,
    outline = OutlineDark,
    outlineVariant = OutlineVariantDark,
    inverseSurface = InverseSurfaceDark,
    inverseOnSurface = InverseOnSurfaceDark,
    inversePrimary = InversePrimaryDark,
    surfaceDim = SurfaceDimDark,
    surfaceBright = SurfaceBrightDark,
    surfaceContainerLowest = SurfaceContainerLowestDark,
    surfaceContainerLow = SurfaceContainerLowDark,
    surfaceContainer = SurfaceContainerDark,
    surfaceContainerHigh = SurfaceContainerHighDark,
    surfaceContainerHighest = SurfaceContainerHighestDark
)

// 统一形状 token：信息卡和设置分组保持克制的 8dp 内圆角。
object QlhShapeTokens {
    val compact = 4.dp
    val control = 8.dp
    val dialog = 12.dp
}

/** Shared low-cost geometry tokens corresponding to the PC surface vocabulary. */
object QlhUiTokens {
    val pageHorizontal = 16.dp
    val sectionGap = 18.dp
    val rowGap = 8.dp
    val hairline = 1.dp
    val cornerMark = 28.dp
    val touchTarget = 48.dp
}

@Composable
fun qlhBrandGold() = if (MaterialTheme.colorScheme.background.luminance() < 0.5f) BrandGoldDark else BrandGoldLight

@Composable
fun qlhBrandGoldContainer() = if (MaterialTheme.colorScheme.background.luminance() < 0.5f) BrandGoldContainerDark else BrandGoldContainerLight

@Composable
fun qlhOnBrandGoldContainer() = if (MaterialTheme.colorScheme.background.luminance() < 0.5f) OnBrandGoldContainerDark else OnBrandGoldContainerLight

@Composable
fun qlhNeonGreen() = if (MaterialTheme.colorScheme.background.luminance() < 0.5f) NeonGreenDark else NeonGreenLight

@Composable
fun qlhOnNeonGreen() = if (MaterialTheme.colorScheme.background.luminance() < 0.5f) OnNeonGreenDark else OnNeonGreenLight

private val QlhShapes = Shapes(
    extraSmall = RoundedCornerShape(QlhShapeTokens.compact),
    small = RoundedCornerShape(QlhShapeTokens.control),
    medium = RoundedCornerShape(QlhShapeTokens.control),
    large = RoundedCornerShape(QlhShapeTokens.control),
    extraLarge = RoundedCornerShape(QlhShapeTokens.dialog)
)

@Composable
fun QlhTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    // 设置状态栏颜色
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            @Suppress("DEPRECATION")
            window.statusBarColor = colorScheme.background.toArgb()
            val insetsController = WindowCompat.getInsetsController(window, view)
            insetsController.isAppearanceLightStatusBars = !darkTheme
            insetsController.isAppearanceLightNavigationBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = QlhTypography,
        shapes = QlhShapes,
        content = content
    )
}
