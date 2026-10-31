package ai.live.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Locale;

/**
 * Voice conversation for the Live AI page: the phone's speech recognizer turns Thai speech into text and
 * the phone's text-to-speech reads the answer aloud. Exposed to the page as window.LiveAIVoice;
 * results come back through window.__liveaiVoice(event, text) with events result / spoken / error.
 */
class VoiceBridge {
    static final int MIC_REQUEST = 7;
    private static final Locale THAI = new Locale("th", "TH");

    private final Activity act;
    private final WebView web;
    private SpeechRecognizer rec;
    private TextToSpeech tts;
    private boolean ttsReady, ttsThai, pendingListen;
    private String pendingSpeech;

    VoiceBridge(Activity act, WebView web) {
        this.act = act;
        this.web = web;
        tts = new TextToSpeech(act, status -> {
            ttsReady = status == TextToSpeech.SUCCESS;
            if (ttsReady) {
                int r = tts.setLanguage(THAI);
                ttsThai = r != TextToSpeech.LANG_MISSING_DATA && r != TextToSpeech.LANG_NOT_SUPPORTED;
                tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override public void onStart(String id) { }
                    @Override public void onDone(String id) { if ("end".equals(id)) send("spoken", ""); }
                    @Override public void onError(String id) { send("spoken", ""); }
                });
            }
            if (pendingSpeech != null) { String t = pendingSpeech; pendingSpeech = null; speak(t); }
        });
    }

    private void send(String event, String text) {
        String js = "window.__liveaiVoice&&window.__liveaiVoice(" + JSONObject.quote(event) + "," + JSONObject.quote(text) + ")";
        act.runOnUiThread(() -> web.evaluateJavascript(js, null));
    }

    @JavascriptInterface
    public void listen() {
        act.runOnUiThread(() -> {
            if (act.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                pendingListen = true;
                act.requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, MIC_REQUEST);
                return;
            }
            startListening();
        });
    }

    /** Called by the activity after the microphone permission dialog. */
    void onMicPermission(boolean granted) {
        if (!pendingListen) return;
        pendingListen = false;
        if (granted) startListening(); else send("error", "permission");
    }

    private void startListening() {
        if (!SpeechRecognizer.isRecognitionAvailable(act)) { send("error", "unavailable"); return; }
        if (tts != null) tts.stop();
        if (rec == null) {
            rec = SpeechRecognizer.createSpeechRecognizer(act);
            rec.setRecognitionListener(new RecognitionListener() {
                @Override public void onResults(Bundle b) {
                    ArrayList<String> r = b.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                    send("result", r == null || r.isEmpty() ? "" : r.get(0));
                }
                @Override public void onError(int code) {
                    String why;
                    switch (code) {
                        case SpeechRecognizer.ERROR_NO_MATCH:
                        case SpeechRecognizer.ERROR_SPEECH_TIMEOUT: why = "no-speech"; break;
                        case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS: why = "permission"; break;
                        case SpeechRecognizer.ERROR_NETWORK:
                        case SpeechRecognizer.ERROR_NETWORK_TIMEOUT: why = "network"; break;
                        default: why = "error-" + code;
                    }
                    send("error", why);
                }
                @Override public void onReadyForSpeech(Bundle b) { }
                @Override public void onBeginningOfSpeech() { }
                @Override public void onRmsChanged(float v) { }
                @Override public void onBufferReceived(byte[] b) { }
                @Override public void onEndOfSpeech() { }
                @Override public void onPartialResults(Bundle b) { }
                @Override public void onEvent(int t, Bundle b) { }
            });
        }
        Intent i = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        i.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        i.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "th-TH");
        i.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "th-TH");
        i.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);
        i.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, act.getPackageName());
        rec.startListening(i);
    }

    @JavascriptInterface
    public void speak(String text) {
        act.runOnUiThread(() -> {
            if (tts == null) { send("spoken", ""); return; }
            if (!ttsReady) { pendingSpeech = text; return; }
            if (!ttsThai) { send("error", "tts-th-missing"); return; }
            tts.stop();
            // the engine has a maximum length per call: queue the answer in pieces split at spaces
            int max = Math.max(200, TextToSpeech.getMaxSpeechInputLength() - 100);
            String rest = text.trim();
            int n = 0;
            while (!rest.isEmpty()) {
                String part = rest;
                if (part.length() > max) {
                    int cut = rest.lastIndexOf(' ', max);
                    if (cut < max / 2) cut = max;
                    part = rest.substring(0, cut);
                }
                rest = rest.substring(part.length()).trim();
                tts.speak(part, n == 0 ? TextToSpeech.QUEUE_FLUSH : TextToSpeech.QUEUE_ADD, null, rest.isEmpty() ? "end" : "p" + n);
                n++;
            }
            if (n == 0) send("spoken", "");
        });
    }

    @JavascriptInterface
    public void stop() {
        act.runOnUiThread(() -> {
            pendingListen = false;
            if (rec != null) rec.cancel();
            if (tts != null) tts.stop();
        });
    }

    void release() {
        if (rec != null) { rec.destroy(); rec = null; }
        if (tts != null) { tts.shutdown(); tts = null; }
    }
}
