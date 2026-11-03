import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "H3TextureLab.WindowPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["H3TextureWindowPreview", "H3TextureStoredWindowPreview"].includes(nodeData.name)) return;
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            previous?.apply(this, arguments);
            const info = message?.h3_texture_window_preview?.[0];
            if (!info) return;
            if (!this._h3WindowPreview) {
                const container = document.createElement("div");
                Object.assign(container.style, { display: "flex", flexDirection: "column", gap: "8px", padding: "8px", overflow: "auto", color: "var(--fg-color, #ddd)", background: "var(--comfy-menu-bg, #252525)" });
                const widget = this.addDOMWidget("current_window", "h3-window-preview", container, { serialize: false });
                widget.computeSize = () => [420, 390];
                this._h3WindowPreview = { container, widget };
                this.setSize([Math.max(this.size[0], 460), Math.max(this.size[1], 640)]);
            }
            const container = this._h3WindowPreview.container;
            for (const video of container.querySelectorAll("video")) { video.pause(); video.removeAttribute("src"); video.load(); }
            container.replaceChildren();
            const caption = document.createElement("div");
            caption.textContent = `Window ${info.window_index + 1}/${info.total_windows} · frames ${info.source_start}–${info.source_end_exclusive - 1} · preview ${info.scale}×`;
            container.append(caption);
            const row = document.createElement("div");
            Object.assign(row.style, { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" });
            for (const [key, title] of [["source", "Source + person mask"], ["generated", "Generated window"]]) {
                if (!info.urls[key]) continue;
                const panel = document.createElement("div");
                const label = document.createElement("div"); label.textContent = title;
                const video = document.createElement("video");
                Object.assign(video, { controls: true, muted: true, loop: true, autoplay: true, playsInline: true });
                Object.assign(video.style, { width: "100%", maxHeight: "320px", objectFit: "contain", background: "#111" });
                video.src = api.apiURL(info.urls[key]);
                panel.append(label, video); row.append(panel);
            }
            container.append(row);
            this.setDirtyCanvas(true, true);
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            for (const video of this._h3WindowPreview?.container.querySelectorAll("video") ?? []) {
                video.pause(); video.removeAttribute("src"); video.load();
            }
            return removed?.apply(this, arguments);
        };
    },
});
