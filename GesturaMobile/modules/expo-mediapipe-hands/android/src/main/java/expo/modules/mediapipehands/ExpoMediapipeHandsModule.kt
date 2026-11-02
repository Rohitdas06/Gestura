package expo.modules.mediapipehands

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
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
 * image from the given path, rotates according to EXIF, runs MediaPipe
 * HandLandmarker to extract 21 normalised 3-D landmarks (63 doubles),
 * and returns them to JS.
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

        // Attempt 1: Load directly from APK assets
        try {
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
            Log.i(TAG, "Successfully initialized HandLandmarker from assets")
            return created
        } catch (e: Throwable) {
            Log.w(TAG, "Could not load hand_landmarker.task from assets, checking internal storage...", e)
        }

        // Attempt 2: Fallback to internal storage / cache directory
        val fallbackFiles = listOf(
            File(context.filesDir, "hand_landmarker.task"),
            File(context.cacheDir, "hand_landmarker.task")
        )
        for (fallbackFile in fallbackFiles) {
            if (fallbackFile.exists() && fallbackFile.length() > 0) {
                try {
                    val baseOptions = BaseOptions.builder()
                        .setModelAssetPath(fallbackFile.absolutePath)
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
                    Log.i(TAG, "Successfully initialized HandLandmarker from fallback file: ${fallbackFile.absolutePath}")
                    return created
                } catch (fbErr: Throwable) {
                    Log.w(TAG, "Fallback initialization failed for ${fallbackFile.absolutePath}", fbErr)
                }
            }
        }

        initFailed = true
        Log.e(TAG, "Failed to initialize HandLandmarker: model file not found in assets or storage")
        return null
    }

    private fun detectFromFile(filePath: String): List<Double> {
        val landmarker = ensureLandmarker() ?: return emptyList()

        val cleanPath = filePath.removePrefix("file://")
        val file = File(cleanPath)
        if (!file.exists()) return emptyList()

        var bitmap: Bitmap? = null
        var rotatedBitmap: Bitmap? = null
        var argbBitmap: Bitmap? = null

        return try {
            // 1. Inspect image dimensions without allocating memory
            val boundsOptions = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFile(file.absolutePath, boundsOptions)
            val origW = boundsOptions.outWidth
            val origH = boundsOptions.outHeight

            // 2. Downsample large multi-megapixel camera photos to ~384-512px for lightning-fast MediaPipe inference
            var sampleSize = 1
            if (origW > 0 && origH > 0) {
                while ((origW / (sampleSize * 2)) >= 384 && (origH / (sampleSize * 2)) >= 384) {
                    sampleSize *= 2
                }
            }

            val options = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            bitmap = BitmapFactory.decodeFile(file.absolutePath, options)
                ?: return emptyList()

            // Handle EXIF orientation (critical for front camera portrait orientation)
            var orientation = ExifInterface.ORIENTATION_NORMAL
            try {
                val exif = ExifInterface(file.absolutePath)
                orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            } catch (_: Throwable) {
                // Ignore EXIF read errors
            }

            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            }

            rotatedBitmap = if (!matrix.isIdentity) {
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }

            argbBitmap = if (rotatedBitmap.config != Bitmap.Config.ARGB_8888) {
                rotatedBitmap.copy(Bitmap.Config.ARGB_8888, false)
            } else {
                rotatedBitmap
            }

            val mpImage = BitmapImageBuilder(argbBitmap).build()
            val result = landmarker.detect(mpImage)

            if (result.landmarks().isEmpty()) {
                emptyList()
            } else {
                val hand = result.landmarks()[0]
                val flat = ArrayList<Double>(63)
                for (lm in hand) {
                    flat.add(lm.x().toDouble())
                    flat.add(lm.y().toDouble())
                    flat.add(lm.z().toDouble())
                }
                flat
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error in detectFromFile", e)
            emptyList()
        } finally {
            try { file.delete() } catch (_: Throwable) {}
            try {
                if (argbBitmap != null && argbBitmap != rotatedBitmap && argbBitmap != bitmap) {
                    argbBitmap.recycle()
                }
                if (rotatedBitmap != null && rotatedBitmap != bitmap) {
                    rotatedBitmap.recycle()
                }
                bitmap?.recycle()
            } catch (_: Throwable) {}
        }
    }

    companion object {
        private const val TAG = "ExpoMediapipeHands"
    }
}
