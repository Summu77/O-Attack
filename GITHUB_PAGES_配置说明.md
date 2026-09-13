# O-Attack 项目页：预览、上传与 GitHub Pages 配置

## 本地查看

当前预览地址：http://127.0.0.1:8765/

也可以直接用浏览器打开 `docs/index.html`。页面没有构建依赖，实验数据通过本地脚本加载，离线也能查看。

重新启动预览：在 `project-page` 目录运行：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

## 上传目录

上传目标：https://github.com/Summu77/O-Attack

将本项目的 `docs` 文件夹放到仓库 `main` 分支的根目录，保留现有代码与 README。目录结构如下：

```text
O-Attack/
├── README.md
├── GITHUB_PAGES_配置说明.md
└── docs/
    ├── .nojekyll
    ├── index.html
    ├── styles.css
    ├── script.js
    └── assets/
        ├── o-attack-paper.pdf
        ├── results.json
        ├── results.js
        └── …图表资源
```

`docs/assets/o-attack-paper.pdf` 使用当前保存的 13 页正文 PDF。网页的作者、机构、方法与实验内容依据当前论文正文整理；没有添加论文录用、DOI 或 arXiv 编号等未经确认的信息。

## 在 GitHub 上启用 Pages

1. 打开 https://github.com/Summu77/O-Attack/settings/pages 。
2. 找到 **Build and deployment**。
3. 将 **Source** 设为 **Deploy from a branch**。
4. 在 **Branch** 中选择 **main**。
5. 将右侧文件夹设为 **/docs**。
6. 点击 **Save**。
7. 等待 Pages 构建结束。可在仓库 **Actions** 中查看 `pages build and deployment` 的执行状态。
8. 构建成功后访问 https://summu77.github.io/O-Attack/ 。

不要选 `/ (root)`，因为网站入口在 `docs/index.html`。也不需要选择 GitHub Actions 作为 Source 或添加自定义工作流。

以上步骤依据 [GitHub 官方 Pages 配置文档](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。

## 后续维护

- 修改介绍文字、作者、页面链接：编辑 `docs/index.html`。
- 修改颜色、间距与手机样式：编辑 `docs/styles.css`。
- 更新实验表：同时更新 `docs/assets/results.json` 和 `docs/assets/results.js`，或者使用本文后面的同步命令。
- 替换论文：更新 `docs/assets/o-attack-paper.pdf`，文件名保持不变。
- 替换图表：更新 `docs/assets/` 中相应图片。
- 新增 arXiv / 正式发表信息：在正式公开后更新论文入口及 BibTeX。

更新 `results.json` 后，可用 Python 同步网页数据脚本：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('docs/assets')
data = json.loads((p / 'results.json').read_text())
(p / 'results.js').write_text('window.OATTACK_RESULTS = ' + json.dumps(data) + ';\n')
PY
```

提交到 `main` 分支后，GitHub Pages 会自动更新网站。如果暂时看到旧版本，可等待构建完成后强制刷新浏览器。
