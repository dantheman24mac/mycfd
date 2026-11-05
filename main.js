const { app, BrowserWindow } = require('electron');
const path = require('path');

// Make sure WebGPU is available and prefer the fastest adapter.
app.commandLine.appendSwitch('enable-unsafe-webgpu');
app.commandLine.appendSwitch('force_high_performance_gpu');

function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 950,
    backgroundColor: '#05070c',
    title: 'MyCFD – 3D Wind Tunnel',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });
  win.loadFile(path.join(__dirname, 'index.html'));
  if (process.env.MYCFD_DEVTOOLS) win.webContents.openDevTools({ mode: 'detach' });

  // Forward renderer console to the terminal (useful for diagnosing GPU errors).
  win.webContents.on('console-message', (e) => {
    const { level, message } = e;
    console.log(`[renderer:${level}] ${message}`);
  });

  // Test hook: MYCFD_SHOT=<file.png> captures a screenshot after a delay and exits.
  if (process.env.MYCFD_SHOT) {
    setTimeout(async () => {
      const img = await win.webContents.capturePage();
      require('fs').writeFileSync(process.env.MYCFD_SHOT, img.toPNG());
      const stats = await win.webContents.executeJavaScript("document.getElementById('stats').textContent + '\\n' + document.getElementById('error').textContent");
      console.log('[stats] ' + stats);
      app.quit();
    }, +(process.env.MYCFD_SHOT_DELAY || 12000));
  }
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
