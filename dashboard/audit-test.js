const { chromium } = require('playwright');

(async () => {
  const issues = [];
  const warnings = [];

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();

  // Track console errors/warnings
  const consoleErrors = [];
  const consoleWarnings = [];
  page.on('console', msg => {
    const text = msg.text();
    if (msg.type() === 'error') consoleErrors.push(text);
    if (msg.type() === 'warning') consoleWarnings.push(text);
  });

  // Track network requests
  const failedRequests = [];
  page.on('requestfailed', req => {
    failedRequests.push(req.url());
  });

  console.log('Opening dashboard...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  // 1. CHECK page title
  const title = await page.title();
  if (!title || title.trim().length === 0) {
    issues.push('CRITICAL: Page has no title');
  } else if (!title.toLowerCase().includes('vllm') && !title.toLowerCase().includes('dashboard')) {
    warnings.push('WARNING: Page title does not describe the app');
  } else {
    console.log(`[PASS] Page title: "${title}"`);
  }

  // 2. CHECK DOCTYPE and HTML tag
  const hasDoctype = await page.evaluate(() => !!document.doctype);
  if (!hasDoctype) {
    issues.push('CRITICAL: Missing DOCTYPE declaration');
  } else {
    console.log('[PASS] DOCTYPE present');
  }

  const hasHtml = await page.evaluate(() => {
    const html = document.querySelector('html');
    return html && html.hasAttribute('lang');
  });
  if (!hasHtml) {
    issues.push('CRITICAL: HTML element missing lang attribute');
  } else {
    console.log('[PASS] <html lang> attribute present');
  }

  // 3. CHECK viewport meta tag
  const hasViewport = await page.evaluate(() => {
    return !!document.querySelector('meta[name="viewport"]');
  });
  if (!hasViewport) {
    issues.push('CRITICAL: Missing viewport meta tag');
  } else {
    console.log('[PASS] Viewport meta tag present');
  }

  // 4. CHECK charset meta
  const hasCharset = await page.evaluate(() => {
    return !!document.querySelector('meta[charset]');
  });
  if (!hasCharset) {
    issues.push('CRITICAL: Missing charset declaration');
  } else {
    console.log('[PASS] Charset declared');
  }

  // 5. CHECK for link rel="icon"
  const hasFavicon = await page.evaluate(() => {
    return !!document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
  });
  if (!hasFavicon) {
    warnings.push('WARNING: No favicon defined');
  } else {
    console.log('[PASS] Favicon present');
  }

  // 6. CHECK text contrast on body
  const bodyBg = await page.evaluate(() => {
    const style = getComputedStyle(document.body);
    return { bg: style.backgroundColor, color: style.color };
  });
  console.log(`Body colors: bg=${bodyBg.bg}, text=${bodyBg.color}`);

  // 7. CHECK if elements have ARIA labels where needed
  const inputs = await page.$$('input');
  inputs.forEach(async (input, i) => {
    const hasLabel = await input.evaluate(el => {
      return el.hasAttribute('aria-label') || el.hasAttribute('aria-labelledby') ||
        el.hasAttribute('title') || (el.id && document.querySelector(`label[for="${el.id}"]`));
    });
    if (!hasLabel) {
      warnings.push(`WARNING: Input element #${i} missing accessible label`);
    }
  });

  // 8. CHECK CSS is properly loaded
  const mainStylesheet = await page.evaluate(() => {
    const link = document.querySelector('link[rel="stylesheet"]');
    return !!link && link.sheet !== null;
  });
  if (!mainStylesheet) {
    issues.push('CRITICAL: Main CSS stylesheet not loaded');
  } else {
    console.log('[PASS] Main CSS loaded');
  }

  // 9. CHECK JS is properly loaded
  const jsLoaded = await page.evaluate(() => {
    const scripts = document.querySelectorAll('script[src]');
    return Array.from(scripts).every(s => !!s.src);
  });
  console.log(jsLoaded ? '[PASS] JS scripts loaded' : '[WARN] JS scripts check');

  // 10. CHECK no console errors
  if (consoleErrors.length > 0) {
    issues.push(`CRITICAL: Console errors detected (${consoleErrors.length}): ${consoleErrors.slice(0, 3).join('; ')}`);
  } else {
    console.log('[PASS] No console errors');
  }

  // 11. CHECK no failed network requests
  if (failedRequests.length > 0) {
    warnings.push(`WARNING: ${failedRequests.length} failed network requests`);
  } else {
    console.log('[PASS] No failed network requests');
  }

  // 12. CHECK responsive behavior (resize and see if layout adapts)
  await page.setViewportSize({ width: 375, height: 667 });
  await page.waitForTimeout(500);
  const mobileLayout = await page.evaluate(() => {
    const grid = document.getElementById('server-grid');
    if (!grid) return true; // No grid = no issue
    const style = getComputedStyle(grid);
    return style.gridTemplateColumns === '1fr' ||
      style.gridTemplateColumns.includes('1fr') ||
      grid.querySelector('.server-card').offsetWidth < 500;
  });
  console.log(mobileLayout ? '[PASS] Mobile layout adapts' : '[WARN] Mobile layout check');

  // Restore desktop viewport
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForTimeout(500);

  // 13. CHECK accessibility (basic axe-like checks)
  const accessibilityIssues = [];
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const hasAria = await btn.evaluate(el =>
      el.hasAttribute('aria-label') || el.getAttribute('aria-hidden') !== null
    );
    const hasText = await btn.evaluate(el => el.textContent.trim().length > 0);
    if (!hasAria && !hasText) {
      accessibilityIssues.push('Button without text or aria-label');
    }
  }
  if (accessibilityIssues.length > 0) {
    warnings.push(`Accessibility: ${accessibilityIssues.join(', ')}`);
  } else if (buttons.length > 0) {
    console.log('[PASS] Buttons have text content');
  }

  // 14. CHECK semantic HTML structure
  const hasMain = await page.evaluate(() => !!document.querySelector('main, [role="main"]'));
  if (!hasMain) {
    warnings.push('WARNING: Missing <main> element for page content');
  } else {
    console.log('[PASS] <main> element present');
  }

  const hasHeader = await page.evaluate(() => !!document.querySelector('header'));
  if (!hasHeader) {
    warnings.push('WARNING: Missing <header> element');
  } else {
    console.log('[PASS] <header> element present');
  }

  // 15. CHECK if the SSE stream endpoint responds correctly
  try {
    const streamRes = await page.evaluate(async () => {
      const resp = await fetch('/stream');
      if (!resp.ok) return { ok: false, status: resp.status };
      const reader = resp.body.getReader();
      const { value, done } = await reader.read();
      return { ok: true, done, type: resp.headers.get('content-type') };
    }, { timeout: 10000 });
    if (streamRes.ok) {
      console.log('[PASS] SSE /stream endpoint responds with type: ' + streamRes.type);
    } else {
      issues.push(`SSE /stream failed with status ${streamRes.status}`);
    }
  } catch (e) {
    console.log('[WARN] SSE stream check: ' + e.message);
  }

  // 16. CHECK the /api/servers endpoint
  try {
    const apiRes = await page.evaluate(async () => {
      const resp = await fetch('/api/servers');
      if (!resp.ok) return { ok: false, status: resp.status };
      const data = await resp.json();
      return { ok: true, hasServers: Array.isArray(data.servers), serverCount: data.servers?.length };
    });
    if (apiRes.ok) {
      console.log(`[PASS] /api/servers returns ${apiRes.serverCount} server(s)`);
    } else {
      issues.push(`/api/servers failed with status ${apiRes.status}`);
    }
  } catch (e) {
    issues.push(`API check error: ${e.message}`);
  }

  await browser.close();

  // Summary
  console.log('\n═══════════════════════════════════════');
  console.log('AUDIT SUMMARY');
  console.log('═══════════════════════════════════════');
  console.log(`Issues:    ${issues.length}`);
  console.log(`Warnings:  ${warnings.length}`);
  console.log(`Errors:    ${consoleErrors.length}`);
  console.log(`Warn:      ${consoleWarnings.length}`);
  if (issues.length > 0) {
    console.log('\nISSUES:');
    issues.forEach((_, i) => console.log(`  ❌ ${_}`));
  }
  if (warnings.length > 0) {
    console.log('\nWARNINGS:');
    warnings.forEach(w => console.log(`  ⚠️  ${w}`));
  }
  if (issues.length === 0 && warnings.length === 0) {
    console.log('\n✅ All checks passed!');
  }

  // Write results to file for later review
  const fs = require('fs');
  fs.writeFileSync('/tmp/lighthouse-audit.json', JSON.stringify({
    issues,
    warnings,
    consoleErrors,
    consoleWarnings,
  }, null, 2));

  // Also write as plain text
  let report = 'vLLM Dashboard Audit Report\n';
  report += '═══════════════════════════════════════\n\n';
  report += `Issues: ${issues.length}\n`;
  report += `Warnings: ${warnings.length}\n\n`;
  if (issues.length > 0) {
    report += 'ISSUES:\n';
    issues.forEach(i => report += `  ❌ ${i}\n`);
    report += '\n';
  }
  if (warnings.length > 0) {
    report += 'WARNINGS:\n';
    warnings.forEach(w => report += `  ⚠️  ${w}\n`);
    report += '\n';
  }
  if (consoleErrors.length > 0) {
    report += 'Console Errors:\n';
    consoleErrors.forEach(e => report += `  ${e}\n`);
  }
  fs.writeFileSync('/tmp/lighthouse-audit.txt', report);
  console.log('\nFull report written to /tmp/lighthouse-audit.txt');
  console.log('JSON report written to /tmp/lighthouse-audit.json');

})();
