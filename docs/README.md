# AdaptiveDiffuseMotion 项目网站

这个文件夹包含了 AdaptiveDiffuseMotion 项目的 GitHub Pages 网站文件。

## 文件说明

- `index.html` - 主页面，包含项目介绍、方法、结果等内容
- `style.css` - 样式文件，定义网站的视觉效果
- `script.js` - JavaScript 文件，添加交互功能
- `images/` - 存放图片、视频等媒体文件（需要创建）

## 自定义说明

### 1. 添加作者和机构信息

在 `index.html` 中修改以下部分：

```html
<div class="authors">
    <span class="author">你的名字</span>
</div>
<div class="affiliations">
    <span class="affiliation">你的研究机构</span>
</div>
```

### 2. 更新项目摘要

在 Abstract 部分添加你的项目详细描述。

### 3. 添加图片和视频

1. 在 `docs/` 文件夹下创建 `images/` 文件夹
2. 将你的图片、GIF、视频文件放入其中
3. 在 HTML 中替换占位符，例如：

```html
<img src="images/architecture.png" alt="模型架构图">
<video src="images/demo.mp4" controls></video>
```

### 4. 添加论文链接

如果你的论文已发表，更新论文按钮的链接：

```html
<a href="https://arxiv.org/your-paper-link" class="btn btn-secondary">论文</a>
```

## 发布到 GitHub Pages

按照以下步骤发布网站：

1. 提交所有文件到 GitHub
2. 进入仓库的 Settings > Pages
3. 在 "Source" 下选择 "Deploy from a branch"
4. 在 "Branch" 下选择 "main" 分支和 "/docs" 文件夹
5. 点击 "Save"
6. 等待几分钟后，你的网站将在 `https://你的用户名.github.io/AdaptiveDiffuseMotion/` 访问

## 本地预览

使用 Python 本地服务器预览：

```bash
cd docs
python -m http.server 8000
```

然后在浏览器访问 `http://localhost:8000`
