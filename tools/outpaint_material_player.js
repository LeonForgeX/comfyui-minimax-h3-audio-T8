"use strict";

// Local evidence only. Blob playback supports seeking even on http.server
// installations that ignore Range. Never label unverified media as ready.
(() => {
  const config = JSON.parse(document.getElementById("review-data").textContent);
  const videos = [document.getElementById("source"), document.getElementById("outpaint")];
  const status = document.getElementById("status");
  const transport = [...document.querySelectorAll("[data-transport]")];
  const scrub = document.getElementById("scrub");
  const lastTime = (config.frames - 1) / 24;
  const urls = [];
  const crops = [];
  document.getElementById("export").disabled = true;
  let ready = false, busy = false, playing = false, broken = false;
  let pending = Promise.resolve();
  const rect = config.source_rectangle;
  const [width, height] = config.output_geometry;
  const band = 48;
  const regions = [
    ["上边接缝", [0, Math.max(0, rect[1] - band), width, Math.min(height, rect[1] + band) - Math.max(0, rect[1] - band)], rect[1] > 0],
    ["下边接缝", [0, Math.max(0, rect[3] - band), width, Math.min(height, rect[3] + band) - Math.max(0, rect[3] - band)], rect[3] < height],
    ["左边接缝", [Math.max(0, rect[0] - band), 0, Math.min(width, rect[0] + band) - Math.max(0, rect[0] - band), height], rect[0] > 0],
    ["右边接缝", [Math.max(0, rect[2] - band), 0, Math.min(width, rect[2] + band) - Math.max(0, rect[2] - band), height], rect[2] < width],
  ];
  for (const [label, region, visible] of regions) {
    if (!visible) continue;
    const card = document.createElement("section");
    const heading = document.createElement("h3");
    heading.textContent = label + "（扩画成片原比例局部，无额外锐化）";
    const canvas = document.createElement("canvas");
    [canvas.width, canvas.height] = region.slice(2);
    canvas.setAttribute("aria-label", label);
    card.append(heading, canvas);
    document.getElementById("crops").append(card);
    crops.push([canvas, region]);
  }
  scrub.max = String(lastTime);
  scrub.step = String(1 / 24);

  function fail(error) {
    broken = true;
    playing = false;
    videos.forEach(v => v.pause());
    transport.forEach(control => control.disabled = true);
    document.getElementById("export").disabled = true;
    status.textContent = "加载/播放检查失败：" + error.message;
    status.dataset.state = "error";
  }

  function paint() {
    const video = videos[1];
    if (video.readyState >= 2) {
      for (const [canvas, region] of crops) {
        canvas.getContext("2d").drawImage(video, ...region, 0, 0, canvas.width, canvas.height);
      }
    }
    if (ready && !broken) {
      scrub.value = String(Math.min(lastTime, video.currentTime));
      status.textContent = `${playing ? "播放中" : "已暂停"}：左 ${videos[0].currentTime.toFixed(3)} 秒 / 右 ${video.currentTime.toFixed(3)} 秒`;
      status.dataset.state = playing ? "playing" : "paused";
    }
  }

  function waitFor(video, event, done) {
    if (done()) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => finish(new Error("视频等待超时：" + event)), 15000);
      function finish(error) {
        clearTimeout(timer);
        video.removeEventListener(event, success);
        video.removeEventListener("error", failure);
        error ? reject(error) : resolve();
      }
      function success() { if (done()) finish(); }
      function failure() { finish(new Error("视频解码失败")); }
      video.addEventListener(event, success);
      video.addEventListener("error", failure);
    });
  }

  function move(time, play = false) {
    pending = pending.then(async () => {
      if (!ready || broken) return;
      busy = true;
      playing = false;
      transport.forEach(control => control.disabled = true);
      try {
        videos.forEach(v => v.pause());
        const target = Math.max(0, Math.min(lastTime, time));
        await Promise.all(videos.map(async video => {
          if (Math.abs(video.currentTime - target) > 0.001) video.currentTime = target;
          await waitFor(video, "seeked", () => !video.seeking && Math.abs(video.currentTime - target) < 0.025);
        }));
        // Seeking completion is required, not just assignment to currentTime.
        paint();
        if (play) {
          await Promise.all(videos.map(v => v.play()));
          playing = true;
        }
        paint();
      } finally {
        busy = false;
        transport.forEach(control => control.disabled = broken);
      }
    }).catch(fail);
    return pending;
  }

  for (const video of videos) {
    video.addEventListener("seeking", () => { if (ready && !busy) move(video.currentTime, playing); });
    video.addEventListener("play", () => { if (ready && !busy && !playing) move(video.currentTime >= lastTime ? 0 : video.currentTime, true); });
    video.addEventListener("pause", () => { if (ready && !busy && playing && video.paused && !video.ended) move(video.currentTime, false); });
    video.addEventListener("ended", () => { if (!busy) move(lastTime, false); });
    video.addEventListener("error", () => fail(new Error("视频解码失败")));
  }
  videos[1].addEventListener("timeupdate", () => {
    paint();
    if (playing && !busy && Math.abs(videos[0].currentTime - videos[1].currentTime) > 0.12) {
      move(videos[1].currentTime, true);
    }
  });
  videos[1].addEventListener("seeked", paint);
  // Video-frame callbacks also keep seam crops current while normal playback runs.
  function frame() {
    paint();
    if (!broken && videos[1].requestVideoFrameCallback) videos[1].requestVideoFrameCallback(frame);
  }
  if (videos[1].requestVideoFrameCallback) videos[1].requestVideoFrameCallback(frame);

  document.getElementById("play").onclick = () => move(videos[1].currentTime >= lastTime - 0.01 ? 0 : videos[1].currentTime, true);
  document.getElementById("restart").onclick = () => move(0, true);
  document.getElementById("pause").onclick = () => move(videos[1].currentTime, false);
  document.querySelectorAll("button[data-time]").forEach(button => button.onclick = () => move(Number(button.dataset.time)));
  scrub.onchange = () => move(Number(scrub.value));

  document.getElementById("export").onclick = () => {
    if (!ready || broken) return;
    const value = id => document.getElementById(id).value;
    const out = {
      schema: config.schema, review_id: config.review_id, exported_at: new Date().toISOString(),
      source_mode: config.source_mode, video_sha256: {source: config.source.sha256, outpaint: config.outpaint.sha256},
      media_hashes_verified: true, review_completed: document.getElementById("viewed").checked,
      overall: value("overall"), motion: value("motion"), seam_color: value("seam"), source_content: value("sourceKeep"), notes: value("notes"),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2) + "\n"], {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = "outpaint_material_review.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  async function load(video, entry, dimensions) {
    const response = await fetch(entry.file, {cache: "no-store"});
    if (!response.ok) throw new Error(`${entry.file}: HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const actual = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join("");
    if (actual !== entry.sha256) throw new Error(entry.file + " 校验值不匹配，拒绝播放旧片或错片");
    const url = URL.createObjectURL(new Blob([bytes], {type: "video/mp4"}));
    urls.push(url);
    video.src = url;
    await waitFor(video, "loadeddata", () => video.readyState >= 2);
    if (video.videoWidth !== dimensions[0] || video.videoHeight !== dimensions[1] || video.duration < lastTime) {
      throw new Error(entry.file + " 尺寸或时长不符合审片记录");
    }
    video.dataset.sha256Verified = actual;
  }
  window.addEventListener("pagehide", () => { videos.forEach(v => v.pause()); urls.forEach(url => URL.revokeObjectURL(url)); });
  Promise.all([
    load(videos[0], config.source, [rect[2] - rect[0], rect[3] - rect[1]]),
    load(videos[1], config.outpaint, config.output_geometry),
  ]).then(async () => {
    if (broken) return;
    ready = true;
    document.getElementById("export").disabled = false;
    status.dataset.mediaVerified = "true";
    await move(0);
  }).catch(fail);
})();
