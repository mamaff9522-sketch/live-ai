package ai.live.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.DownloadManager;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.text.InputType;
import android.view.View;
import android.webkit.CookieManager;
import android.webkit.URLUtil;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.Toast;

/**
 * Live AI for Android: the Live AI web app (chat + video studio) in a full-screen WebView.
 * Adds what a plain browser tab lacks: picking many clips / a ZIP for upload, saving finished
 * videos to Downloads, the back button, and changing the server address if it ever moves.
 */
public class MainActivity extends Activity {
    private static final int PICK_FILES = 1;
    private static final String PREFS = "liveai";

    private WebView web;
    private ProgressBar bar;
    private ValueCallback<Uri[]> pending;
    private boolean errorShown;

    private String serverUrl() {
        String saved = getSharedPreferences(PREFS, MODE_PRIVATE).getString("server", null);
        return (saved == null || saved.isEmpty()) ? BuildConfig.SERVER_URL : saved;
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        FrameLayout root = new FrameLayout(this);
        web = new WebView(this);
        bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        root.addView(web, new FrameLayout.LayoutParams(-1, -1));
        root.addView(bar, new FrameLayout.LayoutParams(-1, 8));
        setContentView(root);

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);          // the sign-in code lives in localStorage
        s.setDatabaseEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setAllowFileAccess(false);
        s.setUserAgentString(s.getUserAgentString() + " LiveAIApp/" + BuildConfig.VERSION_NAME);
        CookieManager.getInstance().setAcceptCookie(true);

        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest r) {
                Uri u = r.getUrl();
                Uri home = Uri.parse(serverUrl());
                // Live AI's own pages stay in the app; its "download" links answer with
                // Content-Disposition: attachment, which arrives in the DownloadListener below.
                if (u.getHost() != null && u.getHost().equals(home.getHost())) return false;
                try { startActivity(new Intent(Intent.ACTION_VIEW, u)); } catch (ActivityNotFoundException ignored) { }
                return true;                   // other sites open in the normal browser
            }

            @Override
            public void onPageStarted(WebView v, String url, Bitmap icon) { errorShown = false; }

            @Override
            public void onReceivedError(WebView v, WebResourceRequest r, WebResourceError e) {
                if (r.isForMainFrame() && !errorShown) {
                    errorShown = true;
                    showOffline(String.valueOf(e.getDescription()));
                }
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView v, int p) {
                bar.setProgress(p);
                bar.setVisibility(p >= 100 ? View.GONE : View.VISIBLE);
            }

            @Override
            public boolean onShowFileChooser(WebView v, ValueCallback<Uri[]> cb, FileChooserParams params) {
                if (pending != null) pending.onReceiveValue(null);
                pending = cb;
                Intent i = new Intent(Intent.ACTION_GET_CONTENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                String[] types = params.getAcceptTypes();
                String type = (types != null && types.length == 1 && !types[0].isEmpty() && !types[0].startsWith(".")) ? types[0] : "*/*";
                i.setType(type);
                if (params.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE) i.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                try {
                    startActivityForResult(Intent.createChooser(i, "เลือกไฟล์"), PICK_FILES);
                } catch (ActivityNotFoundException e) {
                    pending = null;
                    return false;
                }
                return true;
            }
        });
        web.setDownloadListener((url, ua, disposition, mime, length) -> download(url, disposition, mime));

        if (state != null) web.restoreState(state);
        else web.loadUrl(serverUrl());
    }

    /** Saves a file (a finished video, subtitles) to the phone's Download folder. */
    private void download(String url, String disposition, String mime) {
        try {
            DownloadManager.Request req = new DownloadManager.Request(Uri.parse(url));
            String name = URLUtil.guessFileName(url, disposition, mime);
            String ua = web.getSettings().getUserAgentString();
            req.addRequestHeader("User-Agent", ua);
            req.setTitle(name);
            req.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
            req.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, name);
            ((DownloadManager) getSystemService(DOWNLOAD_SERVICE)).enqueue(req);
            Toast.makeText(this, "กำลังดาวน์โหลดไปที่โฟลเดอร์ Download", Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            try { startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url))); } catch (ActivityNotFoundException ignored) { }
        }
    }

    private void showOffline(String why) {
        new AlertDialog.Builder(this)
            .setTitle("เชื่อมต่อ Live AI ไม่ได้")
            .setMessage("ตรวจอินเทอร์เน็ตของมือถือ แล้วลองใหม่\n\nที่อยู่ตอนนี้: " + serverUrl() + "\n(" + why + ")")
            .setPositiveButton("ลองใหม่", (d, w) -> web.loadUrl(serverUrl()))
            .setNeutralButton("เปลี่ยนที่อยู่เซิร์ฟเวอร์", (d, w) -> askServer())
            .setCancelable(false)
            .show();
    }

    private void askServer() {
        EditText in = new EditText(this);
        in.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        in.setText(serverUrl());
        new AlertDialog.Builder(this)
            .setTitle("ที่อยู่เซิร์ฟเวอร์ Live AI")
            .setMessage("เว้นว่างเพื่อกลับไปใช้ที่อยู่เริ่มต้น")
            .setView(in)
            .setPositiveButton("บันทึก", (d, w) -> {
                String u = in.getText().toString().trim();
                if (!u.isEmpty() && !u.startsWith("http")) u = "https://" + u;
                while (u.endsWith("/")) u = u.substring(0, u.length() - 1);
                SharedPreferences.Editor e = getSharedPreferences(PREFS, MODE_PRIVATE).edit();
                if (u.isEmpty()) e.remove("server"); else e.putString("server", u);
                e.apply();
                web.loadUrl(serverUrl());
            })
            .setNegativeButton("ยกเลิก", (d, w) -> web.loadUrl(serverUrl()))
            .show();
    }

    @Override
    protected void onActivityResult(int req, int result, Intent data) {
        if (req != PICK_FILES || pending == null) { super.onActivityResult(req, result, data); return; }
        Uri[] uris = null;
        if (result == RESULT_OK && data != null) {
            if (data.getClipData() != null) {
                int n = data.getClipData().getItemCount();
                uris = new Uri[n];
                for (int i = 0; i < n; i++) uris[i] = data.getClipData().getItemAt(i).getUri();
            } else if (data.getData() != null) {
                uris = new Uri[]{data.getData()};
            }
        }
        pending.onReceiveValue(uris);
        pending = null;
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack(); else super.onBackPressed();
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }
}
