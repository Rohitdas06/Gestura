package expo.modules.mediapipehands

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise
import java.io.File

/**
 * Expo Module that wraps Google MediaPipe HandLandmarker.
 *
 * Provides `detectLandmarksFromFile(filePath)` which loads a JPEG/PNG
 * image from the given path, runs hand landmark detection, and returns
 * the 21 normalised 3-D landmarks (63 doubles) for the first detected hand.
 *
 * The MediaPipe `hand_landmarker.task` model file must be placed in
 * `android/app/src/main/assets/`.
 *
 * HandLandmarker is created lazily on first use. Creating it in OnCreate
 * crashes the whole app if the JNI `.so` is missing (common on emulators)
 * or if the React context is not ready yet.
 */
class ExpoMediapipeHandsModule : Module() {

    private var handLandmarker: HandLandmarker? = null
    private var initFailed = false

    override fun definition() = ModuleDefinition {
        Name("ExpoMediapipeHands")

        OnDestroy {
            try {
                handLandmarker?.close()
            } catch (_: Throwable) {
                // ignore
            }
            handLandmarker = null
        }

        /**
         * detectLandmarksFromFile(filePath) → Promise<number[]>
         *
         * @param filePath  Absolute path to a JPEG/PNG image on-device.
         * @returns 63 doubles (x0,y0,z0,…,x20,y20,z20) or empty array.
         */
        AsyncFunction("detectLandmarksFromFile") { filePath: String, promise: Promise ->
            try {
                val landmarks = detectFromFile(filePath)
                promise.resolve(landmarks)
            } catch (e: Throwable) {
                Log.e(TAG, "detectLandmarksFromFile failed", e)
                promise.resolve(emptyList<Double>())
            }
        }
    }

    @Synchronized
    private fun ensureLandmarker(): HandLandmarker? {
        if (initFailed) return null
        handLandmarker?.let { return it }

        val context: Context = appContext.reactContext
            ?: appContext.currentActivity
            ?: return null

        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            val created = HandLandmarker.createFromOptions(context, options)
            handLandmarker = created
            created
        } catch (e: Throwable) {
            initFailed = true
            Log.e(TAG, "Failed to initialize HandLandmarker", e)
            null
        }
    }

    private fun detectFromFile(filePath: String): List<Double> {
        val landmarker = ensureLandmarker() ?: return emptyList()

        val cleanPath = filePath.removePrefix("file://")
        val file = File(cleanPath)
        if (!file.exists()) return emptyList()

        val options = BitmapFactory.Options().apply {
            inSampleSize = 2
        }
        val bitmap = BitmapFactory.decodeFile(file.absolutePath, options)
            ?: return emptyList()

        val argbBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false).also { bitmap.recycle() }
        } else {
            bitmap
        }

        val mpImage = BitmapImageBuilder(argbBitmap).build()
        val result = landmarker.detect(mpImage)

        file.delete()

        if (result.landmarks().isEmpty()) {
            argbBitmap.recycle()
            return emptyList()
        }

        val hand = result.landmarks()[0]
        val flat = ArrayList<Double>(63)
        for (lm in hand) {
            flat.add(lm.x().toDouble())
            flat.add(lm.y().toDouble())
            flat.add(lm.z().toDouble())
        }

        argbBitmap.recycle()
        return flat
    }

    companion object {
        private const val TAG = "ExpoMediapipeHands"
    }
}
