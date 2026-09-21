// Presentation only: keep the existing project, controls, event handlers and services.
export function createDirectorWorkbench(ctx) {
    const { root, doc, shot, assets, results, videos } = ctx;
    const $ = selector => root.querySelector(selector);
    const $$ = selector => [...root.querySelectorAll(selector)];
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = new URL('./workbench.css', import.meta.url).href;
    document.head.append(css);
    root.dataset.workbench = 'parallel';
    root.setAttribute('aria-label','曜石导演台');
    document.title = '曜石导演台 · 并排创作台';
    let activePanel = null;

    function menu(label, className = '') {
        const node = document.createElement('details');
        node.className = 'w-menu ' + className;
        node.innerHTML = `<summary>${label}</summary><div class="w-menu-body"></div>`;
        return node;
    }
    const header = $('.o-header');
    $('.o-brand h2').textContent = '曜石导演台';
    $('.o-eyebrow').textContent = '并排创作台';
    const project = $('.o-project');
    const projectActions = project.querySelector('.o-row');
    const actions = document.createElement('div');
    actions.className = 'w-header-actions';
    actions.append($('[data-action="model-settings"]'), $('[data-service="save"]'));
    const more = menu('更多');
    const moreBody = more.querySelector('.w-menu-body');
    moreBody.append(...projectActions.children);
    moreBody.append($('[data-action="undo"]'), $('[data-action="redo"]'));
    const oldBadge = $('.o-badge');
    oldBadge.parentElement.remove();
    moreBody.append(oldBadge);
    const note = $('.o-note');
    moreBody.append(note);
    projectActions.remove();
    actions.append(more);
    if (window.parent === window) {
        const back = document.createElement('a');
        back.href = '/'; back.className = 'w-back'; back.textContent = '返回画布';
        actions.append(back);
    }
    header.append(actions);
    const save = $('[data-save]');
    save.title = save.textContent;
    new MutationObserver(() => { save.title = save.textContent; }).observe(save, {childList:true,subtree:true,characterData:true});

    const workspace = $('.o-workspace');
    const stage = $('.o-stage');
    const shots = $('.o-shotbar');
    const shotActions = shots.querySelector('.o-row > .o-row');
    const shotMenu = menu('镜头操作');
    shotMenu.querySelector('.w-menu-body').append($('[data-action="duplicate"]'), $('[data-action="delete-shot"]'));
    const shotFooter = document.createElement('div');
    shotFooter.className = 'w-shot-actions';
    shotFooter.append($('[data-action="new"]'), shotMenu);
    shotActions.remove();
    shots.append(shotFooter);
    shots.querySelector('h3').firstChild.textContent = '镜头列表 ';
    const editor = document.createElement('section');
    editor.className = 'w-editor'; editor.setAttribute('aria-label', '当前镜头剧本');
    editor.innerHTML = '<div class="w-shot-heading"><h2 data-workbench-heading></h2><div data-workbench-name></div></div>';
    for (const selector of ['[data-global-zone]','[data-writing-zone]','[data-simple-zone]','[data-local-zone]']) editor.append($(selector));
    editor.append($('[data-events]').closest('details'));
    const settings = document.createElement('section');
    settings.className = 'w-settings';
    settings.innerHTML = '<h3>基础设置</h3><div class="w-setting-tiles" data-workbench-tiles></div><button class="w-enhancements" data-workbench-panel="d3"><span>增强与性能</span><small data-workbench-enhancements></small></button>';
    settings.append($('[data-inspector]'));
    editor.append(settings);
    workspace.prepend(shots, editor);

    const mediaWorkbench = $('.o-media-workbench');
    const thumbs = $('[data-thumbs]');
    const mediaSection = document.createElement('section');
    mediaSection.className = 'w-assets';
    mediaSection.innerHTML = '<div class="w-assets-heading"><h3>本镜素材 <small data-workbench-asset-count></small></h3><div class="o-row" data-workbench-asset-actions></div></div>';
    const mediaActions = mediaSection.querySelector('[data-workbench-asset-actions]');
    const add = stage.querySelector(':scope > .o-row [data-action="add"]');
    add.textContent = '添加素材';
    mediaActions.append(add);
    const mediaMenu = menu('素材操作');
    mediaMenu.querySelector('.w-menu-body').append($('[data-action="all"]'), $('[data-action="pair"]'), $('[data-action="zoom"]'), $('[data-action="toggle-preview"]'));
    mediaActions.append(mediaMenu);
    mediaSection.append(thumbs, $('.o-trayhelp'), $('[data-insert-target]'));
    mediaMenu.querySelector('.w-menu-body').append(mediaSection.querySelector('.o-trayhelp'));
    const insertTarget=mediaSection.querySelector('[data-insert-target]');
    new MutationObserver(()=>{insertTarget.title=insertTarget.textContent;}).observe(insertTarget,{childList:true,characterData:true,subtree:true});
    const selectedActions=document.createElement('div');
    selectedActions.className='w-selection-actions';
    selectedActions.setAttribute('aria-label','选中素材操作');
    thumbs.after(selectedActions);
    mediaWorkbench.after(mediaSection);
    const tabs = $('.o-viewbar .o-tabs');
    tabs.prepend($('[data-view="output"]'));
    $('[data-view="input"]').textContent = '输入预览';
    const toolsMenu = menu('检查与导出', 'w-tools-menu');
    for (const button of $$('.o-footer [data-service],.o-footer [data-action="check"]')) toolsMenu.querySelector('.w-menu-body').append(button);
    moreBody.prepend(toolsMenu);
    const footerActions = $('.o-footer > .o-row');
    footerActions.prepend($('[data-action="generate-all"]'));
    const mobileShots = document.createElement('button');
    mobileShots.className = 'w-mobile-shots'; mobileShots.textContent = '镜头列表';
    mobileShots.setAttribute('aria-expanded','false');
    mobileShots.addEventListener('click', () => { const open = root.dataset.shotsOpen !== 'true'; root.dataset.shotsOpen = String(open); mobileShots.setAttribute('aria-expanded',String(open)); });
    header.prepend(mobileShots);

    // Global uploads keep their original route after the global editor moves columns.
    const globalZone = $('[data-global-zone]');
    globalZone.addEventListener('dragover', e => { if(e.dataTransfer.types.includes('Files')) e.preventDefault(); });
    globalZone.addEventListener('drop', e => { if(e.dataTransfer.files.length){e.preventDefault();ctx.importFiles(e.dataTransfer.files,{target:'global',shot:shot().id});} });
    root.addEventListener('click', e => {
        const panel = e.target.closest('[data-workbench-panel]');
        if(panel){ activePanel = activePanel === panel.dataset.workbenchPanel ? null : panel.dataset.workbenchPanel; syncPanels(); }
        if(e.target.closest('[data-workbench-close]')){ activePanel = null; syncPanels(); }
        if(e.target.closest('[data-workbench-dismiss]'))$('[data-notice]').hidden=true;
        if(e.target.closest('[data-shot]')) {root.dataset.shotsOpen='false';mobileShots.setAttribute('aria-expanded','false');}
    });
    root.addEventListener('keydown', e => { if(e.key==='Escape'){ $$('.w-menu[open]').forEach(m=>m.open=false); root.dataset.shotsOpen='false'; mobileShots.setAttribute('aria-expanded','false'); } });
    // Existing service handlers stop propagation at root. Close menus before that boundary.
    document.addEventListener('click',e=>{
        const button=e.target.closest('.w-menu button');
        if(button && root.contains(button) && !button.disabled){
            const parents=$$('.w-menu[open]').filter(m=>m.contains(button));
            queueMicrotask(()=>parents.forEach(m=>m.open=false));
        }
        for(const m of $$('.w-menu[open]')) if(!m.contains(e.target))m.open=false;
    },true);

    function syncPanels() {
        $$('[data-workbench-section]').forEach(section => section.hidden=section.dataset.workbenchSection!==activePanel);
        $$('[data-workbench-panel]').forEach(button => button.setAttribute('aria-expanded',String(button.dataset.workbenchPanel===activePanel)));
    }
    function syncInspector() {
        const inspector = $('[data-inspector]');
        const sections = [...inspector.children];
        const name = inspector.querySelector('[data-field="name"]');
        if(name){$('[data-workbench-name]').replaceChildren(name.closest('label'));sections[0].hidden=true;}
        for(const [index,key] of ['source','audio','timing','d3'].entries()){
            const section=sections[index+1]; if(!section)continue;
            section.dataset.workbenchSection=key;
            section.querySelector('h3').insertAdjacentHTML('afterend','<button type="button" class="w-close-panel" data-workbench-close aria-label="收起设置">收起</button>');
        }
        syncSummary(); syncPanels();
    }
    function syncSummary() {
        const s=shot();
        $('[data-workbench-heading]').textContent='第 '+String(doc().shots.indexOf(s)+1).padStart(2,'0')+' 镜';
        const mode={text:'文字',first:'首帧',ends:'首尾',refs:'参考素材'}[s.mode];
        const audio={native:'模型生成声音',record:'音频驱动',voice:'参考音色'}[s.sound];
        const sampling=s.samplingInherit===false?s.sampling:doc().sampling;
        const mp=(sampling?.mode==='two_pass'?sampling.output_mp:sampling?.resolution_mp)??doc().generation?.resolution_mp??'auto';
        const tiles=[['source','画面来源',mode],['audio','声音',audio],['timing','时长',s.duration+' 秒'],['timing','画幅',s.ratio+(doc().sharedRatio?' · 全片共用':' · 本镜')]];
        const host=$('[data-workbench-tiles]');
        if(!host.children.length) host.innerHTML=tiles.map(([key,label])=>`<button data-workbench-panel="${key}"><small>${label}</small><span></span></button>`).join('')+'<button data-action="model-settings"><small>分辨率</small><span></span></button>';
        [...host.children].forEach((button,i)=>{button.querySelector('span').textContent=i<4?tiles[i][2]:mp==='auto'?'自动':mp+' MP';});
        const d=ctx.effectiveD3(s);
        const enabled=[d.semantic_bridge?.enabled&&'语义桥接',d.prompt_relay?.enabled&&'提示词接力',d.fast_h3_v2?.enabled&&'FastH3 V2',d.memory?.low_vram&&'低显存',d.memory?.chunk_ffn&&'分块前馈'].filter(Boolean);
        $('[data-workbench-enhancements]').textContent=enabled.length?enabled.join(' · '):'未启用增强';
        $('[data-action="generate-all"]').textContent='按顺序生成全部（'+doc().shots.length+' 镜）';
        $('[data-global-zone] > summary').innerHTML='全片设定 <small>· '+doc().shots.length+' 镜共用</small>';
        for(const button of $$('[data-writing]'))button.textContent=button.dataset.writing==='simple'?'新手模式':'高级模式';
        syncPanels();
    }
    function syncAssets() {
        const cards=$$('[data-thumbs] [data-tray-id]');
        $('[data-workbench-asset-count]').textContent='（'+cards.length+'）';
        if(cards.length&&!cards.some(c=>c.dataset.selected==='true'))cards[0].dataset.selected='true';
        for(const card of cards){const a=assets().get(card.dataset.trayId);if(a)card.title=a.name;}
        const selected=cards.find(card=>card.dataset.selected==='true');
        if(!selected)selectedActions.replaceChildren();
        else if(selected.querySelector('.o-quick')){
            selectedActions.replaceChildren(selected.querySelector('.o-quick'));
            const extra=menu('更多');
            const body=extra.querySelector('.w-menu-body');
            for(const control of selected.querySelectorAll('.o-dragline,.o-delete-asset'))body.append(control);
            selectedActions.append(extra);
            selected.dataset.workbenchActions='detached';
        }
    }
    function syncShots() {
        root.dataset.shotsOpen='false';
        mobileShots.setAttribute('aria-expanded','false');
        for(const button of $$('[data-shots] [data-shot]')){
            const s=doc().shots.find(s=>s.id===button.dataset.shot);if(!s)continue;
            const a=[s.first,...(s.refs||[]),...(doc().sharedRefs||[]),...(s.tray||[])].map(id=>assets().get(id)).find(a=>a?.kind==='image');
            let thumb=button.querySelector('.w-shot-thumbnail');
            if(a){if(!thumb){thumb=document.createElement('img');thumb.className='w-shot-thumbnail';thumb.alt='';button.prepend(thumb);}if(thumb.getAttribute('src')!==a.url)thumb.src=a.url;}
            else thumb?.remove();
            button.dataset.hasOutput=String(videos(results().get(s.id)).length>0);
        }
    }
    function syncPreview() {
        const player=$('.o-output-player');
        if(player)player.setAttribute('aria-label','第 '+(doc().shots.indexOf(shot())+1)+' 镜上次生成成片');
        const text=$('.o-output-meta span');
        if(text)text.textContent='第 '+(doc().shots.indexOf(shot())+1)+' 镜 · 上次成片';
        const stageEmpty=$('[data-stage] .o-empty p');
        if(stageEmpty&&shot().mode==='text'&&!player)stageEmpty.textContent='在中间提示词区描述画面，无需上传首帧。生成后可在这里回看成片。';
        const selected=$('[data-view][aria-pressed="true"]');
        root.dataset.activeView=selected?.dataset.view||'input';
    }
    return {syncInspector,syncSummary,syncAssets,syncShots,syncPreview};
}
