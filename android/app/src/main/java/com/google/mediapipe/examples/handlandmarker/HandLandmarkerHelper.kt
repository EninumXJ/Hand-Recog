/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.camera2.internal.compat.quirk.AutoFlashUnderExposedQuirk
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.Locale.Category
import kotlin.math.abs

class HandLandmarkerHelper(
    var minHandDetectionConfidence: Float = DEFAULT_HAND_DETECTION_CONFIDENCE,
    var minHandTrackingConfidence: Float = DEFAULT_HAND_TRACKING_CONFIDENCE,
    var minHandPresenceConfidence: Float = DEFAULT_HAND_PRESENCE_CONFIDENCE,
    var maxNumHands: Int = DEFAULT_NUM_HANDS,
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    // this listener is only used when running in RunningMode.LIVE_STREAM
    val handLandmarkerHelperListener: LandmarkerListener? = null
) {

    // For this example this needs to be a var so it can be reset on changes.
    // If the Hand Landmarker will not change, a lazy val would be preferable.
    private var handLandmarker: HandLandmarker? = null

    init {
        setupHandLandmarker()
    }

    fun clearHandLandmarker() {
        handLandmarker?.close()
        handLandmarker = null
    }

    // Return running status of HandLandmarkerHelper
    fun isClose(): Boolean {
        return handLandmarker == null
    }

    // Initialize the Hand landmarker using current settings on the
    // thread that is using it. CPU can be used with Landmarker
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarker
    fun setupHandLandmarker() {
        // Set general hand landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        baseOptionBuilder.setModelAssetPath(MP_HAND_LANDMARKER_TASK)

        // Check if runningMode is consistent with handLandmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (handLandmarkerHelperListener == null) {
                    throw IllegalStateException(
                        "handLandmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Hand Landmarker.
            val optionsBuilder =
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinHandDetectionConfidence(minHandDetectionConfidence)
                    .setMinTrackingConfidence(minHandTrackingConfidence)
                    .setMinHandPresenceConfidence(minHandPresenceConfidence)
                    .setNumHands(maxNumHands)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::returnLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            handLandmarker =
                HandLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            handLandmarkerHelperListener?.onError(
                "Hand Landmarker failed to initialize. See error logs for " +
                        "details"
            )
            Log.e(
                TAG, "MediaPipe failed to load the task with error: " + e
                    .message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            handLandmarkerHelperListener?.onError(
                "Hand Landmarker failed to initialize. See error logs for " +
                        "details", GPU_ERROR
            )
            Log.e(
                TAG,
                "Image classifier failed to load model with error: " + e.message
            )
        }
    }

    // Convert the ImageProxy to MP Image and feed it to HandlandmakerHelper.
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call detectLiveStream" +
                        " while not using RunningMode.LIVE_STREAM"
            )
        }
        val frameTime = SystemClock.uptimeMillis()

        // Copy out RGB bits from the frame to a bitmap buffer
        val bitmapBuffer =
            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                    -1f,
                    1f,
                    imageProxy.width.toFloat(),
                    imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
            matrix, true
        )

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        detectAsync(mpImage, frameTime)
        // HandLandmarker.HandLandmarkerOptions.OutputHandler.ResultListener
    }

    // Run hand hand landmark using MediaPipe Hand Landmarker API
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        handLandmarker?.detectAsync(mpImage, frameTime)

        Log.d("TAG", "handLandmarker: " +handLandmarker);
        // this.returnLivestreamResult()
        Log.d("TAG", "handLandmarkerHelperListener: " +this.handLandmarkerHelperListener);

        // As we're using running mode LIVE_STREAM, the landmark result will
        // be returned in returnLivestreamResult function
    }

    // Accepts the URI for a video file loaded from the user's gallery and attempts to run
    // hand landmarker inference on the video. This process will evaluate every
    // frame in the video and attach the results to a bundle that will be
    // returned.
    fun detectVideoFile(
        videoUri: Uri,
        inferenceIntervalMs: Long
    ): ResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                "Attempting to call detectVideoFile" +
                        " while not using RunningMode.VIDEO"
            )
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        val startTime = SystemClock.uptimeMillis()

        var didErrorOccurred = false

        // Load frames from the video and run the hand landmarker.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong()

        // Note: We need to read width/height from frame instead of getting the width/height
        // of the video directly because MediaRetriever returns frames that are smaller than the
        // actual dimension of the video file.
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height

        // If the video is invalid, returns a null detection result
        if ((videoLengthMs == null) || (width == null) || (height == null)) return null

        // Next, we'll get one frame every frameInterval ms, then run detection on these frames.
        val resultList = mutableListOf<HandLandmarkerResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)

        val frameBufferSize =  5
        var threshold = 2.5
        var gesture = 0

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs // ms

            retriever
                .getFrameAtTime(
                    timestampMs * 1000, // convert from ms to micro-s
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                ?.let { frame ->
                    // Convert the video frame to ARGB_8888 which is required by the MediaPipe
                    val argb8888Frame =
                        if (frame.config == Bitmap.Config.ARGB_8888) frame
                        else frame.copy(Bitmap.Config.ARGB_8888, false)

                    // Convert the input Bitmap object to an MPImage object to run inference
                    val mpImage = BitmapImageBuilder(argb8888Frame).build()

                    // Run hand landmarker using MediaPipe Hand Landmarker API
                    handLandmarker?.detectForVideo(mpImage, timestampMs)
                        ?.let { detectionResult ->
                            resultList.add(detectionResult)
                        } ?: {
                        didErrorOccurred = true
                        handLandmarkerHelperListener?.onError(
                            "ResultBundle could not be returned" +
                                    " in detectVideoFile"
                        )
                    }
                }
                ?: run {
                    didErrorOccurred = true
                    handLandmarkerHelperListener?.onError(
                        "Frame at specified time could not be" +
                                " retrieved when detecting in video."
                    )
                }
            val LandmarkList = mutableListOf<FloatArray>()
            val array3d = Array(frameBufferSize){Array(21){FloatArray(3)}}
            val arrayTwoHands = Array(2){Array(21){FloatArray(3)}}

            /*
                1: x1 - x0 > 0
                2: x1 - x0 < 0
                3: y1 - y0 > 0
                4: y1 - y0 < 0
            */
            if (i > frameBufferSize) {
                Log.d("TAG", "frame");
                // Log.d("TAG", "计算结果：" +resultList.first());
                var handedness = resultList[i.toInt()].handednesses()[0][0].categoryName()
                var numOfHands = resultList[i.toInt()].handednesses().size
                Log.d("TAG", "Handedness：" +handedness);
                Log.d("TAG", "numOfHands：" +numOfHands);
                if (numOfHands == 1){
                    for (j in 0 .. frameBufferSize-1){
                        var frameIndex = j.toInt() + i.toInt() - frameBufferSize + 1;
                        var landMarks = resultList[frameIndex].landmarks()[0];
                        var landMarksSize = landMarks.size;   // 21
                        // Log.d("TAG", "landMarks Size：" +landMarksSize);
                        for (idx in 0..20){
                            array3d[j][idx][0] = landMarks[idx].x() * 512;
                            array3d[j][idx][1] = landMarks[idx].y() * 512;
                            array3d[j][idx][2] = landMarks[idx].z() * 512;

                        }

//                        Log.d("TAG", "frame：" +j.toInt());
                        Log.d("TAG", "worldLandmarks x：" +array3d[j][0][0]);
                        Log.d("TAG", "worldLandmarks y：" +array3d[j][0][1]);
                        // Log.d("TAG", "worldLandmarks z：" +array3d[j][0][2]);
                        // 手从屏幕右侧到左侧 -> x变小（负数）
                    }

                /* Logic Determine:
                    // None:
                        1. No Gesture -- 0
                    // Static:
                        1. Thumb Up   -- 1
                        2. Palm Open  -- 2
                        3. OK         -- 3
                        4. He Shi     -- 4
                    // Dynamic:
                        1. Left       -- 5
                        2. Right      -- 6
                        3. Down       -- 7
                */

                    // 如果只检测到一只手，那么就判断这只手在屏幕上的距离变化
                    if (determineIfStatic(array3d, threshold, frameBufferSize)) {
                        gesture = determineStaticGesture(array3d[frameBufferSize-1])
                    }
                    else{
                        gesture = determineDynamicGesture(array3d, frameBufferSize)
                    }
                }

                else if(numOfHands == 2){
                    if ((resultList[i.toInt()].handednesses()[0][0].categoryName() == "Left" && resultList[i.toInt()].handednesses()[1][0].categoryName() == "Right") ||
                        (resultList[i.toInt()].handednesses()[0][0].categoryName() == "Right" && resultList[i.toInt()].handednesses()[1][0].categoryName() == "Left")  ){
                        var landMarksOne = resultList[i.toInt()].worldLandmarks()[0];
                        var landMarksTwo= resultList[i.toInt()].worldLandmarks()[1];
                        for (idx in 0..20){
                            arrayTwoHands[0][idx][0] = landMarksOne[idx].x() * 256;
                            arrayTwoHands[0][idx][1] = landMarksOne[idx].y() * 256;
                            arrayTwoHands[0][idx][2] = landMarksOne[idx].z() * 256;
                            arrayTwoHands[1][idx][0] = landMarksTwo[idx].x() * 256;
                            arrayTwoHands[1][idx][1] = landMarksTwo[idx].y() * 256;
                            arrayTwoHands[1][idx][2] = landMarksTwo[idx].z() * 256;

                        }

                        gesture = determineStaticGestureTwoHands(arrayTwoHands)
                    }
                }
            }
            // debug
            var gestureCategory = ""
            if (gesture == 0)
                gestureCategory = "No gesture"
            if (gesture == 1)
                gestureCategory = "Thumb Up"
            if (gesture == 2)
                gestureCategory = "Paml Open"
            if (gesture == 3)
                gestureCategory = "OK"
            if (gesture == 4)
                gestureCategory = "He shi"
            if (gesture == 5)
                gestureCategory = "Go Left"
            if (gesture == 6)
                gestureCategory = "Go Right"
            if (gesture == 7)
                gestureCategory = "Go Down"
            Log.d("TAG", "Gesture：" +gestureCategory);
        }

        retriever.release()

        val inferenceTimePerFrameMs =
            (SystemClock.uptimeMillis() - startTime).div(numberOfFrameToRead)

        return if (didErrorOccurred) {
            null
        } else {
            ResultBundle(resultList, inferenceTimePerFrameMs, height, width, gesture)
        }
    }

    fun determineIfStatic(array3d: Array<Array<FloatArray>>, threshold: Double, frameBufferSize: Int): Boolean {
        var flag: Boolean = false;    // dynamic gesture default

        if(abs(array3d[frameBufferSize-1][0][0] - array3d[0][0][0]) < threshold &&
            abs(array3d[frameBufferSize-1][0][1] - array3d[0][0][1]) < threshold){
            flag = true
        }
        var flag_temp: Boolean = false;
        for (j in 1 .. frameBufferSize-1){
            if(abs(array3d[j-1][0][0] - array3d[j][0][0]) < threshold &&
                abs(array3d[j-1][0][1] - array3d[j][0][1]) < threshold ){    // x and y
                flag_temp = true
            }
            else{
                flag_temp = false
            }
            flag = flag && flag_temp
        }

        return flag;
    }

    fun determineDynamicGesture(array3d: Array<Array<FloatArray>>, frameBufferSize: Int): Int {
        var gesture = 0;    // no gesture default
        var leftFlag: Boolean = true
        var rightFlag: Boolean = true
        var downFlag: Boolean = true
        var tempLeftFlag: Boolean = false
        var tempRightFlag: Boolean = false
        var tempDownFlag: Boolean = false
        for (j in 1 .. frameBufferSize-1){
            if (array3d[j][0][0] < array3d[j-1][0][0]){   // left
                tempLeftFlag = true
            }
            if (array3d[j][0][0] >= array3d[j-1][0][0]){   // right
                tempRightFlag = true
            }
            if (array3d[j][0][1] < array3d[j-1][0][1]){    // down
                tempDownFlag = true
            }
            leftFlag = leftFlag && tempLeftFlag
            rightFlag = rightFlag && tempRightFlag
            downFlag = downFlag && tempDownFlag
            tempLeftFlag = false
            tempRightFlag = false
            tempDownFlag = false
        }
        if (leftFlag){
            gesture = 5
        }
        if (rightFlag){
            gesture = 6
        }
        if (downFlag){
            gesture = 7
        }
        return gesture;
    }

    fun determineStaticGesture(array2d: Array<FloatArray>): Int {
        var gesture = 0;    // no gesture default
        var threOK = 0.3
        var threThumbUp = 0.35
        var threPalmOpen = 0.95
        // OK
        if (abs(array2d[8][0]-array2d[4][0]) + abs(array2d[8][1]-array2d[4][1]) + abs(array2d[8][2]-array2d[4][2]) < threOK &&
            abs(array2d[12][0]-array2d[8][0]) + abs(array2d[12][1]-array2d[8][1]) + abs(array2d[12][2]-array2d[8][2]) > threOK &&
            abs(array2d[16][0]-array2d[8][0]) + abs(array2d[16][1]-array2d[8][1]) + abs(array2d[16][2]-array2d[8][2]) > threOK &&
            abs(array2d[20][0]-array2d[8][0]) + abs(array2d[20][1]-array2d[8][1]) + abs(array2d[20][2]-array2d[8][2]) > threOK ){
            gesture = 3
        }
        // Thumb Up
        if (abs(array2d[7][0]-array2d[5][0]) + abs(array2d[7][1]-array2d[5][1]) + abs(array2d[7][2]-array2d[5][2]) < threThumbUp &&
            abs(array2d[11][0]-array2d[9][0]) + abs(array2d[11][1]-array2d[9][1]) + abs(array2d[11][2]-array2d[9][2]) < threThumbUp &&
            abs(array2d[15][0]-array2d[13][0]) + abs(array2d[15][1]-array2d[13][1]) + abs(array2d[15][2]-array2d[13][2]) < threThumbUp &&
            abs(array2d[19][0]-array2d[17][0]) + abs(array2d[19][1]-array2d[17][1]) + abs(array2d[19][2]-array2d[17][2]) < threThumbUp &&
            array2d[4][1] > array2d[3][1] && array2d[3][1] > array2d[2][1] ){
            gesture = 1
        }
        // Palm Open
        // 判断手指是否伸直，即判断手指关节5、6的方向向量和5、7， 5、8组成的向量之间的差距
        var vec_56 = Array(3, {i -> array2d[6][i]-array2d[5][i]}); var vec_57 = Array(3, {i -> array2d[7][i]-array2d[5][i]}); var vec_58 = Array(3, {i -> array2d[8][i]-array2d[5][i]})
        var vec_910 = Array(3, {i -> array2d[10][i]-array2d[9][i]}); var vec_911 = Array(3, {i -> array2d[11][i]-array2d[9][i]}); var vec_912 = Array(3, {i -> array2d[12][i]-array2d[9][i]})
        var vec_1314 = Array(3, {i -> array2d[14][i]-array2d[13][i]}); var vec_1315 = Array(3, {i -> array2d[15][i]-array2d[13][i]}); var vec_1316 = Array(3, {i -> array2d[16][i]-array2d[13][i]})
        var vec_1718 = Array(3, {i -> array2d[18][i]-array2d[17][i]}); var vec_1719 = Array(3, {i -> array2d[19][i]-array2d[17][i]}); var vec_1720 = Array(3, {i -> array2d[20][i]-array2d[17][i]})
        var vec_12 = Array(3, {i -> array2d[2][i]-array2d[1][i]}); var vec_13 = Array(3, {i -> array2d[3][i]-array2d[1][i]}); var vec_14 = Array(3, {i -> array2d[4][i]-array2d[1][i]});
        if (computeSimilarity(vec_56, vec_57) >= threPalmOpen && computeSimilarity(vec_56, vec_58) >= threPalmOpen &&
            computeSimilarity(vec_910, vec_911) >= threPalmOpen && computeSimilarity(vec_910, vec_912) >= threPalmOpen &&
            computeSimilarity(vec_1314, vec_1315) >= threPalmOpen && computeSimilarity( vec_1314, vec_1316) >= threPalmOpen &&
            computeSimilarity(vec_1718, vec_1719) >= threPalmOpen && computeSimilarity(vec_1718, vec_1720) >= threPalmOpen &&
            computeSimilarity(vec_12, vec_13) >= threPalmOpen && computeSimilarity(vec_12, vec_14) >= threPalmOpen){
            gesture = 2
        }
        return gesture;
    }

    fun determineStaticGestureTwoHands(array: Array<Array<FloatArray>>): Int {
        var gesture = 0;
        var leftArray = array[0]
        var rightArray = array[1]
        var thre = 5
        Log.d("TAG", "4 distance x：" +abs(leftArray[4][0]-rightArray[4][0]));
        Log.d("TAG", "4 distance y：" +abs(leftArray[4][1]-rightArray[4][1]));
        Log.d("TAG", "8 distance x：" +abs(leftArray[8][0]-rightArray[8][0]));
        Log.d("TAG", "8 distance y：" +abs(leftArray[8][1]-rightArray[8][1]));
        Log.d("TAG", "12 distance x：" +abs(leftArray[12][0]-rightArray[12][0]));
        Log.d("TAG", "12 distance y：" +abs(leftArray[12][1]-rightArray[12][1]));
        Log.d("TAG", "16 distance x：" +abs(leftArray[16][0]-rightArray[16][0]));
        Log.d("TAG", "16 distance y：" +abs(leftArray[16][1]-rightArray[16][1]));
        Log.d("TAG", "20 distance x：" +abs(leftArray[20][0]-rightArray[20][0]));
        Log.d("TAG", "20 distance y：" +abs(leftArray[20][1]-rightArray[20][1]));
        if (//abs(leftArray[1][0]-rightArray[1][0]) < thre && abs(leftArray[1][1]-rightArray[1][1]) < thre &&  // 1 x-x y-y
            abs(leftArray[4][0]-rightArray[4][0]) < thre && abs(leftArray[4][1]-rightArray[4][1]) < thre &&
            abs(leftArray[8][0]-rightArray[8][0]) < thre && abs(leftArray[8][1]-rightArray[8][1]) < thre &&
            abs(leftArray[12][0]-rightArray[12][0]) < thre && abs(leftArray[12][1]-rightArray[12][1]) < thre &&
            abs(leftArray[16][0]-rightArray[16][0]) < thre && abs(leftArray[16][1]-rightArray[16][1]) < thre &&
            abs(leftArray[20][0]-rightArray[20][0]) < thre && abs(leftArray[20][1]-rightArray[20][1]) < thre)
            gesture = 4
        return gesture
    }
    fun computeSimilarity(vec1: Array<Float>, vec2: Array<Float>): Double{
        var vecDot = Math.sqrt(vec1[0].toDouble() * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])
        var vec1Len = Math.sqrt(vec1[0].toDouble() * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2])
        var vec2Len = Math.sqrt(vec2[0].toDouble() * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2])
        var sim = vecDot / (vec1Len * vec2Len)
        return sim
    }

    // Accepted a Bitmap and runs hand landmarker inference on it to return
    // results back to the caller
    fun detectImage(image: Bitmap): ResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                "Attempting to call detectImage" +
                        " while not using RunningMode.IMAGE"
            )
        }


        // Inference time is the difference between the system time at the
        // start and finish of the process
        val startTime = SystemClock.uptimeMillis()

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(image).build()

        // Run hand landmarker using MediaPipe Hand Landmarker API
        handLandmarker?.detect(mpImage)?.also { landmarkResult ->
            val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
            return ResultBundle(
                listOf(landmarkResult),
                inferenceTimeMs,
                image.height,
                image.width,
                0
            )
        }

        // If handLandmarker?.detect() returns null, this is likely an error. Returning null
        // to indicate this.
        handLandmarkerHelperListener?.onError(
            "Hand Landmarker failed to detect."
        )
        return null
    }

    // Return the landmark result to this HandLandmarkerHelper's caller
    private fun returnLivestreamResult(
        result: HandLandmarkerResult,
        input: MPImage
    ) {
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - result.timestampMs()
        Log.d("TAG", "result: " + result);
        handLandmarkerHelperListener?.onResults(
            ResultBundle(
                listOf(result),
                inferenceTime,
                input.height,
                input.width,
                0
            )
        )
    }

    // Return errors thrown during detection to this HandLandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        handLandmarkerHelperListener?.onError(
            error.message ?: "An unknown error has occurred"
        )
    }

    companion object {
        const val TAG = "HandLandmarkerHelper"
        private const val MP_HAND_LANDMARKER_TASK = "hand_landmarker.task"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_HAND_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_HAND_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_HAND_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_HANDS = 1
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
    }

    data class ResultBundle(
        val results: List<HandLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        val Gesture: Int
    )

    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}
