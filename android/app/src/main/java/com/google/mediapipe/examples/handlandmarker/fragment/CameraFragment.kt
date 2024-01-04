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
package com.google.mediapipe.examples.handlandmarker.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.mediapipe.examples.handlandmarker.HandLandmarkerHelper
import com.google.mediapipe.examples.handlandmarker.MainViewModel
import com.google.mediapipe.examples.handlandmarker.R
import com.google.mediapipe.examples.handlandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.concurrent.withLock
import kotlin.math.abs
import androidx.camera.core.ExperimentalLensFacing
import android.hardware.camera2.CaptureRequest
import android.util.Range

class CameraFragment : Fragment(), HandLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Hand Landmarker"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var handLandmarkerHelper: HandLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    @SuppressLint("UnsafeOptInUsageError")
    private var cameraFacing = CameraSelector.LENS_FACING_EXTERNAL
    // private var cameraFacing = CameraSelector.LENS_FACING_FRONT

    private val frameBufferSize =  4
    private var numOfHands = 0
    private var handness = "None"
    private var counter = 0
    private var arrayTwoHands = Array(2){Array(21){FloatArray(3)}}
    private val array3d:MutableList<Array<FloatArray>> = mutableListOf()
//    private var landmarkerResult: HandLandmarkerHelper.ResultBundle? = null
    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the HandLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if (handLandmarkerHelper.isClose()) {
                handLandmarkerHelper.setupHandLandmarker()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if(this::handLandmarkerHelper.isInitialized) {
            viewModel.setMaxHands(handLandmarkerHelper.maxNumHands)
            viewModel.setMinHandDetectionConfidence(handLandmarkerHelper.minHandDetectionConfidence)
            viewModel.setMinHandTrackingConfidence(handLandmarkerHelper.minHandTrackingConfidence)
            viewModel.setMinHandPresenceConfidence(handLandmarkerHelper.minHandPresenceConfidence)
            viewModel.setDelegate(handLandmarkerHelper.currentDelegate)

            // Close the HandLandmarkerHelper and release resources
            backgroundExecutor.execute { handLandmarkerHelper.clearHandLandmarker() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Create the HandLandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            handLandmarkerHelper = HandLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minHandDetectionConfidence = viewModel.currentMinHandDetectionConfidence,
                minHandTrackingConfidence = viewModel.currentMinHandTrackingConfidence,
                minHandPresenceConfidence = viewModel.currentMinHandPresenceConfidence,
                maxNumHands = viewModel.currentMaxHands,
                currentDelegate = viewModel.currentDelegate,
                handLandmarkerHelperListener = this)
        }

        // Attach listeners to UI control widgets
        initBottomSheetControls()
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings
        fragmentCameraBinding.bottomSheetLayout.maxHandsValue.text =
            viewModel.currentMaxHands.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandPresenceConfidence
            )

        // When clicked, lower hand detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandDetectionConfidence >= 0.2) {
                handLandmarkerHelper.minHandDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandDetectionConfidence <= 0.8) {
                handLandmarkerHelper.minHandDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower hand tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandTrackingConfidence >= 0.2) {
                handLandmarkerHelper.minHandTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandTrackingConfidence <= 0.8) {
                handLandmarkerHelper.minHandTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower hand presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandPresenceConfidence >= 0.2) {
                handLandmarkerHelper.minHandPresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandPresenceConfidence <= 0.8) {
                handLandmarkerHelper.minHandPresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, reduce the number of hands that can be detected at a
        // time
        fragmentCameraBinding.bottomSheetLayout.maxHandsMinus.setOnClickListener {
            if (handLandmarkerHelper.maxNumHands > 1) {
                handLandmarkerHelper.maxNumHands--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of hands that can be detected
        // at a time
        fragmentCameraBinding.bottomSheetLayout.maxHandsPlus.setOnClickListener {
            if (handLandmarkerHelper.maxNumHands < 2) {
                handLandmarkerHelper.maxNumHands++
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        handLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "HandLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset Handlandmarker
    // helper.
    private fun updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxHandsValue.text =
            handLandmarkerHelper.maxNumHands.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandPresenceConfidence
            )

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        backgroundExecutor.execute {
            handLandmarkerHelper.clearHandLandmarker()
            handLandmarkerHelper.setupHandLandmarker()
        }
        fragmentCameraBinding.overlay.clear()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
//        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
//            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
//            .build()
        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            //.setTargetResolution(Size(240, 320))
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work

//        imageAnalyzer =
//            ImageAnalysis.Builder()
//                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
//                //.setTargetResolution(Size(240,320))
//                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
//                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        val builder = ImageAnalysis.Builder()
        val ext: Camera2Interop.Extender<*> = Camera2Interop.Extender(builder)
        ext.setCaptureRequestOption(
            CaptureRequest.CONTROL_AE_MODE,
            CaptureRequest.CONTROL_AE_MODE_OFF
        )
        ext.setCaptureRequestOption(
            CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
            Range<Int>(10, 10)
        )
        builder.setTargetAspectRatio(AspectRatio.RATIO_4_3)
        //builder.setTargetResolution(Size(240,320))
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        val imageAnalyzer = builder.build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        detectHand(image)
                    }
                }
        // update frame buffer
        // updateFrameBuffer()
        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectHand(imageProxy: ImageProxy) {
        handLandmarkerHelper.detectLiveStream(
            imageProxy = imageProxy,
            isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT
        )
    }

    private fun updateFrameBuffer(resultBundle: HandLandmarkerHelper.ResultBundle){
        var array2d = Array(21){FloatArray(3)}
        var resultSize = resultBundle!!.results.size
        Log.d("TAG", "resultSize: " +resultSize);
        var landmarks = resultBundle!!.results.first().worldLandmarks()
        numOfHands = resultBundle!!.results.first().worldLandmarks().size
        Log.d("TAG", "world landmarks: " +landmarks);
        Log.d("TAG", "num of hands: " +numOfHands);

        if (landmarks.size != 0){    // 检测到有手
            if(numOfHands == 1){
                handness = resultBundle!!.results.first().handednesses()[0][0].categoryName()
                Log.d("TAG", "handness: " +handness);
                for (idx in 0..20){
                    array2d[idx][0] = landmarks[0][idx].x() * 512;
                    array2d[idx][1] = landmarks[0][idx].y() * 512;
                    array2d[idx][2] = landmarks[0][idx].z() * 512;

                }
                if (array3d.size == frameBufferSize){
                    array3d.removeAt(0)   // remove the first frame
                }
                if (array3d.size < frameBufferSize){
                    array3d.add(array2d)
                }
            }
            else if(numOfHands == 2){
                var headness1 = resultBundle!!.results.first().handednesses()[0][0].categoryName()
                var headness2 = resultBundle!!.results.first().handednesses()[1][0].categoryName()
                Log.d("TAG", "The First hand: " +headness1);
                Log.d("TAG", "The Second hand: " +headness2);
                if((headness1 == "Left" && headness2 == "Left") || (headness1 == "Left" && headness2 == "Left")){
                    array3d.clear()   // 检测到另一个人的手、清空之前的缓存
                }
                else{
                    for (idx in 0..20){
                        arrayTwoHands[0][idx][0] = landmarks[0][idx].x() * 512;
                        arrayTwoHands[0][idx][1] = landmarks[0][idx].y() * 512;
                        arrayTwoHands[0][idx][2] = landmarks[0][idx].z() * 512;
                        arrayTwoHands[1][idx][0] = landmarks[1][idx].x() * 512;
                        arrayTwoHands[1][idx][1] = landmarks[1][idx].y() * 512;
                        arrayTwoHands[1][idx][2] = landmarks[1][idx].z() * 512;
                    }
                }
            }
        }
        else{     // 没检测到有手,清空所有缓存
            array3d.clear()
        }
    }

    private fun determineGesture(): Int{
        var gesture = 0
        var thresholdIfStatic = 3.5
        var flagIfStatic = determineIfStatic(thresholdIfStatic)   // if static
        if(flagIfStatic){
            Log.d("TAG", "STATIC gesture!");
            gesture = determineStaticGesture()
        }
        else{
            gesture = determineDynamicGesture()
        }
        return gesture
    }

    private fun determineIfStatic(threshold: Double): Boolean{
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

    private fun determineStaticGesture(): Int{
        var gesture = 0;    // no gesture default
        var threOK1 = 40
        var threPalmOpen = 0.85
        var threOK2 = 0.8
        var array2d = array3d[frameBufferSize-1]   // get current frame
        // OK
        var dist1 = abs(array2d[8][0]-array2d[4][0]) + abs(array2d[8][1]-array2d[4][1]) + abs(array2d[8][2]-array2d[4][2])
        Log.d("TAG", "dist1: "+dist1);
        Log.d("TAG", "handness: " +handness);
//        Log.d("TAG", "array2d[8][0]: " +array2d[8][0]);
//        Log.d("TAG", "array2d[7][0]: " +array2d[7][0]);
//        Log.d("TAG", "array2d[4][1]: " +array2d[4][1]);
//        Log.d("TAG", "array2d[2][1]: " +array2d[2][1]);
        var vec_56 = Array(3, { i -> array2d[6][i] - array2d[5][i] });
        var vec_57 = Array(3, { i -> array2d[7][i] - array2d[5][i] });
        var vec_58 = Array(3, { i -> array2d[8][i] - array2d[5][i] })
        var vec_910 = Array(3, { i -> array2d[10][i] - array2d[9][i] });
        var vec_911 = Array(3, { i -> array2d[11][i] - array2d[9][i] });
        var vec_912 = Array(3, { i -> array2d[12][i] - array2d[9][i] })
        var vec_1314 = Array(3, { i -> array2d[14][i] - array2d[13][i] });
        var vec_1315 = Array(3, { i -> array2d[15][i] - array2d[13][i] });
        var vec_1316 = Array(3, { i -> array2d[16][i] - array2d[13][i] })
        var vec_1718 = Array(3, { i -> array2d[18][i] - array2d[17][i] });
        var vec_1719 = Array(3, { i -> array2d[19][i] - array2d[17][i] });
        var vec_1720 = Array(3, { i -> array2d[20][i] - array2d[17][i] })
        var vec_23 = Array(3, { i -> array2d[3][i] - array2d[2][i] });
        var vec_24 = Array(3, { i -> array2d[4][i] - array2d[2][i] });
        var vec_34 = Array(3, { i -> array2d[4][i] - array2d[3][i] });

        if (abs(array2d[8][0]-array2d[4][0]) + abs(array2d[8][1]-array2d[4][1]) + abs(array2d[8][2]-array2d[4][2]) < threOK1 &&
            abs(array2d[12][0]-array2d[8][0]) + abs(array2d[12][1]-array2d[8][1]) + abs(array2d[12][2]-array2d[8][2]) > threOK1 &&
            abs(array2d[16][0]-array2d[8][0]) + abs(array2d[16][1]-array2d[8][1]) + abs(array2d[16][2]-array2d[8][2]) > threOK1 &&
            abs(array2d[20][0]-array2d[8][0]) + abs(array2d[20][1]-array2d[8][1]) + abs(array2d[20][2]-array2d[8][2]) > threOK1 &&
            computeSimilarity(vec_910, vec_911) > threOK2  && computeSimilarity(vec_56, vec_57) > threOK2 &&
            computeSimilarity(vec_1314, vec_1315) > threOK2 && computeSimilarity(vec_1718, vec_1719) > threOK2){
            gesture = 3
        }
        // Thumb Up
        else if ( handness == "Right"  &&
            array2d[8][0] > array2d[7][0] && array2d[8][0] > array2d[6][0] && array2d[8][0] < array2d[0][0]  &&
            array2d[12][0] > array2d[11][0] && array2d[12][0] > array2d[10][0] && array2d[12][0] < array2d[0][0]  &&
            array2d[16][0] > array2d[15][0] && array2d[16][0] > array2d[14][0] && array2d[16][0] < array2d[0][0]  &&
            array2d[20][0] > array2d[19][0] && array2d[20][0] > array2d[18][0] && array2d[20][0] < array2d[0][0]  &&
            array2d[4][1] < array2d[3][1] && array2d[3][1] < array2d[2][1] ){
            gesture = 1
        }
        else if ( handness == "Left"  &&
            array2d[8][0] < array2d[7][0] && array2d[8][0] < array2d[6][0] && array2d[8][0] > array2d[0][0]  &&
            array2d[12][0] < array2d[11][0] && array2d[12][0] < array2d[10][0] && array2d[12][0] > array2d[0][0]  &&
            array2d[16][0] < array2d[15][0] && array2d[16][0] < array2d[14][0] && array2d[16][0] > array2d[0][0]  &&
            array2d[20][0] < array2d[19][0] && array2d[20][0] < array2d[18][0] && array2d[20][0] > array2d[0][0]  &&
            array2d[4][1] < array2d[3][1] && array2d[3][1] < array2d[2][1] ){
            gesture = 1
        }
        // Palm Open
        // 判断手指是否伸直，即判断手指关节5、6的方向向量和5、7， 5、8组成的向量之间的差距
        else {

//        var sim5_67 = computeSimilarity(vec_56, vec_57);  Log.d("TAG", "sim5_67: "+sim5_67);
//        var sim5_68 = computeSimilarity(vec_56, vec_58); Log.d("TAG", "sim5_68: "+sim5_68);
//        var sim9_1011 = computeSimilarity(vec_910, vec_911); Log.d("TAG", "sim9_1011: "+sim9_1011);
//        var sim9_1012 = computeSimilarity(vec_910, vec_912); Log.d("TAG", "sim9_1012: "+sim9_1012);
//        var sim13_1415 = computeSimilarity(vec_1314, vec_1315); Log.d("TAG", "sim13_1415: "+sim13_1415);
//        var sim13_1416 = computeSimilarity( vec_1314, vec_1316); Log.d("TAG", "sim13_1416: "+sim13_1416);
//        var sim17_1819 = computeSimilarity(vec_1718, vec_1719); Log.d("TAG", "sim17_1819: "+sim17_1819);
//        var sim17_1820 = computeSimilarity(vec_1718, vec_1720); Log.d("TAG", "sim17_1820: "+sim17_1820);
//        var sim2_34 = computeSimilarity(vec_23, vec_24); Log.d("TAG", "sim2_34: "+sim2_34);
//        var sim23_34 = computeSimilarity(vec_23, vec_34); Log.d("TAG", "sim23_34: "+sim23_34);
            if (computeSimilarity(vec_56, vec_57) >= threPalmOpen && computeSimilarity(
                    vec_56,
                    vec_58
                ) >= threPalmOpen &&
                computeSimilarity(vec_910, vec_911) >= threPalmOpen && computeSimilarity(
                    vec_910,
                    vec_912
                ) >= threPalmOpen &&
                computeSimilarity(vec_1314, vec_1315) >= threPalmOpen && computeSimilarity(
                    vec_1314,
                    vec_1316
                ) >= threPalmOpen &&
                computeSimilarity(vec_1718, vec_1719) >= threPalmOpen && computeSimilarity(
                    vec_1718,
                    vec_1720
                ) >= threPalmOpen &&
                computeSimilarity(vec_23, vec_24) >= threPalmOpen && computeSimilarity(
                    vec_23,
                    vec_34
                ) >= threPalmOpen
            ) {
                gesture = 2
            }
        }
        return gesture;
    }

    fun computeSimilarity(vec1: Array<Float>, vec2: Array<Float>): Double{
        var vecDot = vec1[0].toDouble() * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
        var vec1Len = Math.sqrt(vec1[0].toDouble() * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2])
        var vec2Len = Math.sqrt(vec2[0].toDouble() * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2])

        var sim = vecDot / (vec1Len * vec2Len)
        return sim
    }

    private fun determineDynamicGesture(): Int{
        var gesture = 0;    // no gesture default
        var leftFlag: Boolean = true
        var rightFlag: Boolean = true
        var downFlag: Boolean = true
        var tempLeftFlag: Boolean = false
        var tempRightFlag: Boolean = false
        var tempDownFlag: Boolean = false
        for (j in 1 .. frameBufferSize-1){
            if (array3d[j][0][0] < array3d[j-1][0][0]){   // right
                tempRightFlag = true
            }
            if (array3d[j][0][0] >= array3d[j-1][0][0]){   // left
                tempLeftFlag = true
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

    private fun determineStaticGestureTwoHands(): Int{
        var gesture = 0
        var twoHandsThre = 30;
        if(abs(arrayTwoHands[0][4][0] - arrayTwoHands[1][4][0]) + abs(arrayTwoHands[0][4][1] - arrayTwoHands[1][4][1]) + abs(arrayTwoHands[0][4][2] - arrayTwoHands[1][4][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][8][0] - arrayTwoHands[1][8][0]) + abs(arrayTwoHands[0][8][1] - arrayTwoHands[1][8][1]) + abs(arrayTwoHands[0][8][2] - arrayTwoHands[1][8][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][12][0] - arrayTwoHands[1][12][0]) + abs(arrayTwoHands[0][12][1] - arrayTwoHands[1][12][1]) + abs(arrayTwoHands[0][12][2] - arrayTwoHands[1][12][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][16][0] - arrayTwoHands[1][16][0]) + abs(arrayTwoHands[0][16][1] - arrayTwoHands[1][16][1]) + abs(arrayTwoHands[0][16][2] - arrayTwoHands[1][16][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][20][0] - arrayTwoHands[1][20][0]) + abs(arrayTwoHands[0][20][1] - arrayTwoHands[1][20][1]) + abs(arrayTwoHands[0][20][2] - arrayTwoHands[1][20][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][5][0] - arrayTwoHands[1][5][0]) + abs(arrayTwoHands[0][5][1] - arrayTwoHands[1][5][1]) + abs(arrayTwoHands[0][5][2] - arrayTwoHands[1][5][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][9][0] - arrayTwoHands[1][9][0]) + abs(arrayTwoHands[0][9][1] - arrayTwoHands[1][9][1]) + abs(arrayTwoHands[0][9][2] - arrayTwoHands[1][9][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][13][0] - arrayTwoHands[1][13][0]) + abs(arrayTwoHands[0][13][1] - arrayTwoHands[1][13][1]) + abs(arrayTwoHands[0][13][2] - arrayTwoHands[1][13][2]) < twoHandsThre &&
            abs(arrayTwoHands[0][17][0] - arrayTwoHands[1][17][0]) + abs(arrayTwoHands[0][17][1] - arrayTwoHands[1][17][1]) + abs(arrayTwoHands[0][17][2] - arrayTwoHands[1][17][2]) < twoHandsThre)
            gesture = 4
        return gesture
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after hand have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: HandLandmarkerHelper.ResultBundle
    ) {
        updateFrameBuffer(resultBundle)
        var gesture = 0
        if (numOfHands == 2)
            gesture = determineStaticGestureTwoHands()
        else if(numOfHands == 1){
            if (array3d.size == frameBufferSize){
                gesture = determineGesture()
            }
        }
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                    String.format("%d ms", resultBundle.inferenceTime)

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    //640,
                    resultBundle.inputImageWidth,
                    //480,
                    gesture,
                    RunningMode.LIVE_STREAM
                )

                // Force a redraw
                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            if (errorCode == HandLandmarkerHelper.GPU_ERROR) {
                fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                    HandLandmarkerHelper.DELEGATE_CPU, false
                )
            }
        }
    }
}
