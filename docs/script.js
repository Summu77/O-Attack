(() => {
  'use strict';
  const data = window.OATTACK_RESULTS;
  let group = 'commercial';
  let metric = 0;
  const number = value => metric === 0 ? value.toFixed(1) + '%' : value.toFixed(3);
  function renderResults() {
    const values = data.averages[group];
    const baseline = Math.max(...values.filter((_, i) => i % 2 === metric && i < 12));
    const improvement = values[12 + metric] - baseline;
    document.getElementById('chart-label').textContent = metric === 0 ? 'Average attack success rate (%) ↑' : 'Average semantic similarity ↑';
    document.getElementById('chart-gain').textContent = (metric === 0 ? '+' + improvement.toFixed(1) + ' pp' : '+' + improvement.toFixed(3)) + ' over the strongest baseline';
    document.getElementById('metric-note').textContent = metric === 0 ? 'ASR is the percentage of samples with GPTScore > 0.5. Higher values indicate stronger attacks.' : 'AvgSim is the mean GPTScore across samples. Higher values indicate stronger semantic alignment with the target.';
    const chart = document.getElementById('benchmark-chart');
    chart.replaceChildren(...data.methods.map((method, i) => {
      const row = document.createElement('div');
      row.className = 'chart-row' + (i === 6 ? ' ours' : '');
      const label = document.createElement('span');
      label.className = 'method-name'; label.textContent = method;
      const track = document.createElement('span'); track.className = 'bar-track'; track.setAttribute('aria-hidden', 'true');
      const bar = document.createElement('span'); bar.className = 'bar'; bar.style.width = (values[i * 2 + metric] * (metric === 0 ? 1 : 100)) + '%'; track.append(bar);
      const value = document.createElement('span'); value.textContent = number(values[i * 2 + metric]);
      row.append(label, track, value); return row;
    }));
    const table = document.getElementById('model-table');
    const head = document.createElement('tr');
    ['Model', ...data.methods].forEach((name, i) => { const th = document.createElement('th'); th.scope = 'col'; th.textContent = name; if (i === 7) th.className = 'best'; head.append(th); });
    table.tHead.replaceChildren(head);
    const rows = [...data[group], {model: 'Average (' + data[group].length + ')', values}];
    table.tBodies[0].replaceChildren(...rows.map((item, n) => {
      const row = document.createElement('tr'); if (n === rows.length - 1) row.className = 'average-row';
      const label = document.createElement('th'); label.scope = 'row'; label.textContent = item.model; row.append(label);
      data.methods.forEach((_, i) => { const cell = document.createElement('td'); cell.textContent = number(item.values[i * 2 + metric]); if (i === 6) cell.className = 'best'; row.append(cell); });
      return row;
    }));
    document.getElementById('model-table-caption').textContent = (group === 'commercial' ? '10 frontier commercial MLLMs' : '14 widely used MLLMs') + ' · ' + (metric === 0 ? 'ASR (%) ↑' : 'AvgSim ↑') + ' · Paper, Table 1';
  }
  document.querySelectorAll('[data-group]').forEach(button => button.addEventListener('click', () => {
    group = button.dataset.group;
    document.querySelectorAll('[data-group]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    renderResults();
  }));
  document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => {
    metric = Number(button.dataset.metric);
    document.querySelectorAll('[data-metric]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    renderResults();
  }));
  if (data) renderResults();

  const dialog = document.getElementById('figure-dialog');
  document.querySelectorAll('[data-enlarge]').forEach(button => button.addEventListener('click', () => {
    const img = document.getElementById('enlarged-figure');
    img.src = button.dataset.enlarge; img.alt = button.querySelector('img').alt;
    document.getElementById('figure-title').textContent = button.dataset.title;
    dialog.showModal(); document.body.classList.add('modal-open');
  }));
  document.getElementById('close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
  dialog.addEventListener('close', () => { document.body.classList.remove('modal-open'); });

  document.getElementById('copy-citation').addEventListener('click', async () => {
    const text = document.getElementById('bibtex').textContent;
    const status = document.getElementById('copy-status');
    const button = document.getElementById('copy-citation');
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied ✓'; status.textContent = 'Citation copied to clipboard.';
      setTimeout(() => { button.textContent = 'Copy citation ⧉'; }, 2500);
    } catch {
      const selection = window.getSelection(); const range = document.createRange(); range.selectNodeContents(document.getElementById('bibtex')); selection.removeAllRanges(); selection.addRange(range);
      status.classList.remove('visually-hidden'); status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });
  if ('IntersectionObserver' in window) {
    const navLinks = [...document.querySelectorAll('.nav-links a')];
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) navLinks.forEach(link => { if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); });
    }), {rootMargin: '-15% 0px -55% 0px'});
    navLinks.forEach(link => observer.observe(document.querySelector(link.hash)));
  }
})();
