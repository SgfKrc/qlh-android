# QLH Android UI 改进说明

> 复核日期：2026-08-17
>
> 验证状态：UX-04 至 UX-06 的 Full/Lite Debug APK、Kotlin、androidTest 源集编译及 JVM 单测均已通过（`174` tests）；APK 安装、真机目视、触控、IME 与系统栏验收仍待完成。

本次改动仅涉及 UI 层（`ui/` 目录、`MainActivity.kt` 的导航样式），不改变任何
ViewModel 状态字段、回调、数据层与服务层逻辑。Full / Lite 两个 flavor 通用。

## 一、整体设计

- **配色**：使用中性黑白 Material 3 色调体系：浅色白底黑字、深色黑底白字；
  `surfaceContainer` 系列只承担灰阶层级，绿色、橙色和红色只用于成功、警告和错误状态，
  不承担主题主色。卡片、底栏、输入框靠色调分层，而非阴影。
- **字体**：完善 M3 排版比例（display / headline / title / body / label 全档），
  标题加粗、正文行高放宽、标签统一 Medium 字重；全部 token 字距为 `0sp`，不随控件状态改变。
- **圆角**：统一形状 token：紧凑尾角 `4dp`、控件和普通卡片 `8dp`、对话框 `12dp`。
  仅状态角标保留胶囊形状。
- **共享组件**（新文件 `ui/components/Common.kt`）：
  - `QlhTopBar` 页面大标题顶栏（纯布局实现，不依赖实验性 TopAppBar API）
  - `StatusChip` 状态胶囊角标（本地/远程、连接结果、当前会话）
  - `SettingsGroup` 设置分组卡片（分区标题 + 圆角卡片）
  - `SettingRow` 标准设置行（图标 + 标题/副标题 + 尾部控件）
  - `EmptyState` 空状态占位（图标圆底 + 标题 + 说明）
- 未启用动态取色（dynamic color）：为保证品牌配色在所有设备一致呈现，
  且避免 Android 12 以下回退分叉；如需要可后续在 `Theme.kt` 中加入。

## 二、各页面改动

### 对话页 ChatScreen
- 新增顶栏：会话标题 + 推理模式角标（本地推理 / 远程推理）。
- 气泡重做：去掉 emoji 头像；用户气泡主色底、右对齐，助手气泡容器色、左对齐；
  最大宽度约 85%（对侧固定留白 48dp）；尾侧圆角 `4dp`、其余 `8dp`。
- 生成指标（引擎 / tok/s）与时间戳弱化为小号标签，不再抢视觉焦点。
- 输入栏：填充式 `8dp` 圆角输入框（未聚焦无描边），圆形发送按钮，
  仍保留 imePadding / navigationBarsPadding。
- 空状态改为图标 + 「开始新的对话」（移除硬编码的旧模型名文案）。

### 会话页 SessionListScreen
- 新增顶栏（标题 + 会话总数副标题）。
- 列表卡片：圆角图标底、标题 + 「时间 · 消息数」预览行；
  当前会话高亮为 primaryContainer 且带「当前」角标。
- 新建入口改为带文字的 ExtendedFloatingActionButton「新建会话」。
- 删除仍为卡片尾部图标按钮 + 确认对话框（保留原交互，仅重样式）。

### 设置页 SettingsScreen（重点整理）
- 重组为 8 个分组卡片，顺序：**外观 / 主节点连接 / 推理模式 / 模型管理 /
  推理参数 / 设备状态 / 日志管理 / 关于**，每组统一「分区标题 + 圆角卡片」。
- 模型管理从「推理模式」卡片内部提升为独立分组（显示条件不变：
  完整版且本地推理模式）。
- 设备状态整理为：设备状态（快照 + 系统/内存/存储）、GPU、llama.cpp 后端、
  模型状态、上下文/KV 五个分组卡片，键值行右对齐、分隔线弱化。
- 推理模式切换对话框改为 RadioButton 选项行（原为两个文字按钮）。
- 连接测试结果改为「已连接 / 无法连接」状态角标。
- 每一项设置、回调、testTag（`settings_screen`、`settings_theme_*`）、
  Lite/Full 分支逻辑、日志查看/搜索/复制/分享/清理功能全部保留。
- 滚动容器仍为 `Column + verticalScroll`（结构不变）。

### 底部导航 MainActivity
- NavigationBar 使用 `surfaceContainerLow` 灰阶、零色调抬升和常显标签；
  已明确映射选中/未选中的图标、文字和指示器颜色，并使系统状态栏与导航栏随亮暗主题同步。
- 保留将 `inferenceMode` 传入 ChatScreen（仅复用已有 uiState 字段）。

## 三、构建与自查清单

```bash
cd android
./gradlew assembleFullDebug     # 完整版
./gradlew assembleLiteDebug     # 极简版
./gradlew :app:compileFullDebugKotlin :app:compileLiteDebugKotlin \
  :app:compileFullDebugAndroidTestKotlin :app:compileLiteDebugAndroidTestKotlin \
  :app:testFullDebugUnitTest :app:testLiteDebugUnitTest
```

安装后建议逐项检查：

1. 浅色 / 深色 / 跟随系统三种主题下各页面观感（设置 → 外观切换）。
2. 对话页：发送消息后用户/助手气泡颜色与对齐、长文本气泡不超过约 85% 宽度、
   复制按钮、指标与时间戳、生成中「思考中…」气泡、发送失败重试。
3. 对话页顶栏角标随推理模式切换（全无 → 远程推理 / 全有 → 本地推理）。
4. 会话页：新建、切换、删除会话；当前会话高亮与「当前」角标。
5. 设置页从上到下滚动一遍：连接测试、模式切换对话框、模型目录选择/扫描/
   删除、三个滑杆、设备状态刷新、日志查看（含搜索）/复制/分享/清理、关于。
6. Lite 版：推理模式固定「极简版」、日志区仅复制/分享，均正常。

### UX-05 设置与管理信息架构（2026-08-17）

- 设备状态快照移到设置页首屏，连接和推理模式紧随其后；外观、模型管理、推理参数和日志继续保留原有 Full/Lite 条件与回调。
- 设备系统/内存/存储、GPU、llama.cpp 后端、模型详情、上下文/KV 改为默认收起的详情区，点击后才展开；模型管理只先展示当前模型和数量，完整模型列表按需展开。
- 已接入原有 `showThinking` / `onShowThinkingChange`，显示思考过程改为推理参数分组内的行内 `Switch`，不改变保存字段或对话行为。
- 新增 `CollapsibleSettingsGroup` 共享组件，所有可展开区提供稳定 `testTag`，不新增 API、网络请求或数据层依赖。

### UX-06 开发侧视觉与构建回归（2026-08-17）

- Web 的 `1280x960` 与 `390x844` 截图、无横向溢出和键盘可达性已由 Playwright 覆盖；其结果不作为 Android 真机证据。
- Android 重新执行 Full/Lite Debug 组装、Full/Lite androidTest Kotlin 源集编译与 Full/Lite JVM 单测，均通过；聊天输入框占位文本限制为单行省略，避免窄宽度下抢占输入区。
- 真机验收仍需按下述清单覆盖亮/暗/跟随系统、触控、滚动、软键盘/IME、状态栏和导航栏，不得以编译或 Web 截图关闭该门。

## 四、需要留意的风险点

- 主题色板使用了 material3 的 `surfaceContainer*` 色槽（要求 material3 ≥ 1.2；
  工程已在用 `HorizontalDivider`，属同版本引入，正常应可编译）。
- 输入框填充色使用 `OutlinedTextFieldDefaults.colors(...ContainerColor)` 参数
  （material3 ≥ 1.1）。
- 深色模式下的卡片与背景只允许通过 `surfaceContainer*` 灰阶拉开层次；
  不以绿色或设备动态取色代替主题对比。
