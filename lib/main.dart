import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/material.dart' as material;
import 'dart:ui';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:vector_math/vector_math_64.dart' as vmath;
import 'dart:math' as math;

// Server URLs for emulator access
final String serverUrl = "http://10.0.2.2:8550";
final String wsUrl = "ws://10.0.2.2:8550/ws";

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Pose Estimation with ONNX',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: PoseEstimationPage(cameras: cameras),
    );
  }
}

class PoseEstimationPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const PoseEstimationPage({Key? key, required this.cameras}) : super(key: key);

  @override
  _PoseEstimationPageState createState() => _PoseEstimationPageState();
}

class _PoseEstimationPageState extends State<PoseEstimationPage> {
  late CameraController _cameraController;
  bool _isCameraInitialized = false;

  late OrtSession _session;
  bool _isModelLoaded = false;

  // Change from single list to list of lists for multi-person
  List<List<Offset>> _allArticulationVectors = [];
  // Model input and output sizes
  static const int inputSize = 256; // Model input size (256x256)
  static const int outputSize = 32; // Model output size (e.g., 32x32)
  bool _isProcessing = false;

  // Track current camera lens direction
  CameraLensDirection _currentLensDirection = CameraLensDirection.back;

  // Add a field to store latest YOLO bounding boxes
  List<Rect> _latestYoloBBoxes = [];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  Future<void> _initializeCamera() async {
    try {
      final camera = widget.cameras.firstWhere(
        (camera) => camera.lensDirection == _currentLensDirection,
        orElse: () => widget.cameras.first,
      );
      _cameraController = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController.initialize();
      if (!mounted) return;

      _cameraController.startImageStream(_processCameraImage);
      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  Future<void> _loadModel() async {
    try {
      final rawAssetFile = await rootBundle.load('assets/models/model_opt_S_optimized.onnx');
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(bytes, OrtSessionOptions());
      setState(() {
        _isModelLoaded = true;
      });
      print('ONNX model loaded successfully');
    } catch (e) {
      print('Error loading ONNX model: $e');
    }
  }

  void connectToServer() {
    final uri = Uri.parse(serverUrl);
    // Implement HTTP requests here if needed
  }

  void connectToWebSocket() {
    final channel = WebSocketChannel.connect(Uri.parse(wsUrl));
    // Implement WebSocket communication here
  }

  // Replace _keypoints with _allArticulationVectors update in _processCameraImage
  Future<void> _processCameraImage(CameraImage cameraImage) async {
    if (!_isCameraInitialized || !_isModelLoaded || _isProcessing) return;
    _isProcessing = true;
    try {
      img.Image? image = _convertYUV420ToImage(cameraImage);
      if (image == null) {
        _isProcessing = false;
        return;
      }
      // Log image size
      print('Camera image size: \\${image.width} x \\${image.height}');
      // Check aspect ratio
      final previewSize = _cameraController.value.previewSize;
      if (previewSize != null) {
        final camAR = image.width / image.height;
        final previewAR = previewSize.width / previewSize.height;
        print('Camera AR: \\${camAR.toStringAsFixed(2)}, Preview AR: \\${previewAR.toStringAsFixed(2)}');
        if ((camAR - previewAR).abs() > 0.05) {
          print('WARNING: Camera and preview aspect ratios do not match!');
        }
      }
      List<Rect> personBBoxes = await _detectPersons(image);
      // Store for overlay
      _latestYoloBBoxes = personBBoxes;
      print('Detected \\${personBBoxes.length} person bounding boxes');
      for (final bbox in personBBoxes) {
        print('  BBox: left=\\${bbox.left}, top=\\${bbox.top}, width=\\${bbox.width}, height=\\${bbox.height}');
      }
      List<List<Offset>> articulationVectors = [];
      for (Rect bbox in personBBoxes) {
        img.Image patch = _extractPersonPatch(image, bbox, inputSize);
        vmath.Matrix3 affine = computeAffineTransform(bbox, inputSize);
        OrtValue inputTensor = _prepareInputTensor(patch);
        final outputs = await _session.runAsync(OrtRunOptions(), {'input': inputTensor});
        if (outputs == null || outputs.isEmpty || outputs[0] == null) continue;
        final value = outputs[0]!.value;
        print('Raw model output: \\${value}');
        List<double> outputData;
        if (value is List<double>) {
          outputData = value;
        } else if (value is List && value.isNotEmpty && value[0] is List) {
          if (value[0] is List<double>) {
            outputData = value.expand((e) => e as List<double>).toList();
          } else if (value[0] is List && (value[0] as List).isNotEmpty && (value[0] as List)[0] is List<double>) {
            outputData = value.expand((e) => (e as List).expand((f) => f as List<double>)).toList();
          } else {
            continue;
          }
        } else {
          continue;
        }
        final Float32List outputDataFloat = Float32List.fromList(outputData);
        List<Keypoint> keypoints = _parseKeypointsFromOutput(outputDataFloat);
        vmath.Matrix3 invAffine = invertAffineTransform(affine);
        // In _processCameraImage, ensure articulationVector is defined and used
        List<Offset> articulationVector = mapKeypointsToOriginal(keypoints, invAffine, inputSize);
        articulationVectors.add(articulationVector);
        inputTensor.release();
        for (var ortValue in outputs) {
          ortValue?.release();
        }
      }
      if (articulationVectors.isEmpty) {
        // Fallback: draw a test skeleton in the center
        print('No persons detected, drawing test skeleton.');
        List<Offset> testSkeleton = [
          for (int i = 0; i < 17; i++)
            Offset(image.width / 2 + 30 * (i - 8), image.height / 2)
        ];
        articulationVectors.add(testSkeleton);
      }
      if (mounted) {
        setState(() {
          _allArticulationVectors = articulationVectors;
          // _latestYoloBBoxes already updated
        });
      }
    } catch (e) {
      print("Error processing camera image: $e");
    } finally {
      _isProcessing = false;
    }
  }

  OrtValue _prepareInputTensor(img.Image image) {
    // Ensure the image is 256x256
    final int inputSize = 256;
    img.Image resized = img.copyResize(image, width: inputSize, height: inputSize);

    // Prepare Float32List in NCHW order: [1, 3, 256, 256]
    Float32List inputBuffer = Float32List(1 * 3 * inputSize * inputSize);
    int offset = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          int value = (c == 0)
              ? pixel.r.toInt()
              : (c == 1)
                  ? pixel.g.toInt()
                  : pixel.b.toInt();
          inputBuffer[offset++] = (value - 127.5) / 127.5;
        }
      }
    }

    // Create ONNX input tensor
    return OrtValueTensor.createTensorWithDataList(
      inputBuffer,
      [1, 3, inputSize, inputSize],
    );
  }

  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    try {
      final int width = cameraImage.width;
      final int height = cameraImage.height;

      img.Image image = img.Image(width: width, height: height);

      final Uint8List yPlane = cameraImage.planes[0].bytes;
      final Uint8List uPlane = cameraImage.planes[1].bytes;
      final Uint8List vPlane = cameraImage.planes[2].bytes;
      final int uvRowStride = cameraImage.planes[1].bytesPerRow;
      final int uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 1;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
          final int Y = yPlane[y * width + x];
          final int U = uPlane[uvIndex];
          final int V = vPlane[uvIndex];

          // Convert YUV to RGB
          int r = (Y + (1.370705 * (V - 128))).round().clamp(0, 255);
          int g = (Y - (0.337633 * (U - 128)) - (0.698001 * (V - 128))).round().clamp(0, 255);
          int b = (Y + (1.732446 * (U - 128))).round().clamp(0, 255);

          image.setPixelRgb(x, y, r, g, b);
        }
      }

      return image;
    } catch (e) {
      print('Error converting YUV to RGB: $e');
      return null;
    }
  }

  List<Keypoint> _parseKeypointsFromOutput(Float32List outputData) {
    // If your model outputs 17 or 18 keypoints, adjust here
    int numKeypoints = outputData.length ~/ 3;
    List<Keypoint> keypoints = [];
    for (int i = 0; i < numKeypoints; i++) {
      double y = outputData[i * 3 + 0];
      double x = outputData[i * 3 + 1];
      double confidence = outputData[i * 3 + 2];
      keypoints.add(Keypoint(x: x, y: y, confidence: confidence));
    }
    return keypoints;
  }

  // Helper sigmoid function
  double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  // Non-Maximum Suppression (NMS)
  List<Rect> _nms(List<Map<String, dynamic>> boxes, double iouThreshold) {
    boxes.sort((a, b) => b['conf'].compareTo(a['conf']));
    List<Rect> keep = [];
    List<bool> suppressed = List.filled(boxes.length, false);
    for (int i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;
      final boxA = boxes[i]['rect'] as Rect;
      keep.add(boxA);
      for (int j = i + 1; j < boxes.length; j++) {
        if (suppressed[j]) continue;
        final boxB = boxes[j]['rect'] as Rect;
        final iou = _boxIoU(boxA, boxB);
        if (iou > iouThreshold) suppressed[j] = true;
      }
    }
    return keep;
  }

  double _boxIoU(Rect a, Rect b) {
    final double xA = math.max(a.left, b.left);
    final double yA = math.max(a.top, b.top);
    final double xB = math.min(a.right, b.right);
    final double yB = math.min(a.bottom, b.bottom);
    final double interArea = math.max(0, xB - xA) * math.max(0, yB - yA);
    final double boxAArea = a.width * a.height;
    final double boxBArea = b.width * b.height;
    return interArea / (boxAArea + boxBArea - interArea + 1e-6);
  }

  // YOLOv8 ONNX raw output postprocessing (for 640x640, 3 heads)
  List<Rect> _parseYoloOutput(List<OrtValue?> outputs) {
    if (outputs.isEmpty || outputs[0] == null) return [];
    final value = outputs[0]!.value;
    List raw;
    if (value is List && value.isNotEmpty && value[0] is List) {
      raw = value[0];
    } else {
      return [];
    }
    // Debug: print first 5 predictions and auto-detect numClasses
    print('YOLO raw predictions (first 5):');
    int predLen = raw.isNotEmpty && raw[0] is List ? (raw[0] as List).length : 0;
    int numClasses = predLen > 5 ? predLen - 5 : 0;
    print('Prediction vector length: \\${predLen}, numClasses auto-detected: \\${numClasses}');
    for (int i = 0; i < math.min(5, raw.length); i++) {
      print('  pred[$i]: \\${raw[i]}');
      if (raw[i] is List && numClasses > 0) {
        List<double> classScores = [for (int j = 0; j < numClasses; j++) _sigmoid((raw[i][5 + j] as num).toDouble())];
        int maxClass = 0;
        double maxScore = classScores[0];
        for (int j = 1; j < classScores.length; j++) {
          if (classScores[j] > maxScore) {
            maxScore = classScores[j];
            maxClass = j;
          }
        }
        print('    maxClass: \\${maxClass}, maxScore: \\${maxScore.toStringAsFixed(3)}');
      }
    }
    // YOLOv8: [x, y, w, h, obj_conf, class0, class1, ...]
    // Only keep class 0 (person) by default
    double confThreshold = 0.1; // Lowered for debug
    double iouThreshold = 0.45;
    int classId = 0; // Adjust if person is not class 0
    print('YOLO postprocess: numClasses=\\$numClasses, classId=\\$classId, confThreshold=\\$confThreshold');
    int stride = 4 + 1 + numClasses;
    List<Map<String, dynamic>> boxes = [];
    for (var pred in raw) {
      if (pred is List && pred.length == stride) {
        double x = pred[0].toDouble();
        double y = pred[1].toDouble();
        double w = pred[2].toDouble();
        double h = pred[3].toDouble();
        double objConf = _sigmoid(pred[4].toDouble());
        List<double> classScores = [for (int i = 0; i < numClasses; i++) _sigmoid(pred[5 + i].toDouble())];
        double classConf = classScores[classId];
        double conf = objConf * classConf;
        if (conf > confThreshold) {
          double left = (x - w / 2).clamp(0, 640);
          double top = (y - h / 2).clamp(0, 640);
          double right = (x + w / 2).clamp(0, 640);
          double bottom = (y + h / 2).clamp(0, 640);
          boxes.add({'rect': Rect.fromLTRB(left, top, right, bottom), 'conf': conf});
        }
      }
    }
    // NMS
    List<Rect> finalBoxes = _nms(boxes, iouThreshold);
    print('Parsed YOLO boxes: \\${finalBoxes.length}');
    for (final b in finalBoxes) {
      print('  Box: left=\\${b.left}, top=\\${b.top}, width=\\${b.width}, height=\\${b.height}');
    }
    return finalBoxes;
  }

  img.Image _extractPersonPatch(img.Image image, Rect bbox, int inputSize) {
    img.Image cropped = img.copyCrop(
      image,
      x: bbox.left.toInt(),
      y: bbox.top.toInt(),
      width: bbox.width.toInt(),
      height: bbox.height.toInt(),
    );
    img.Image resized = img.copyResize(cropped, width: inputSize, height: inputSize);
    return resized;
  }

  vmath.Matrix3 computeAffineTransform(Rect bbox, int inputSize) {
    double scaleX = inputSize / bbox.width;
    double scaleY = inputSize / bbox.height;
    double tx = -bbox.left * scaleX;
    double ty = -bbox.top * scaleY;
    return vmath.Matrix3(
      scaleX, 0, tx,
      0, scaleY, ty,
      0, 0, 1,
    );
  }

  vmath.Matrix3 invertAffineTransform(vmath.Matrix3 m) => m.clone()..invert();

  Offset applyInverseAffine(Offset pt, vmath.Matrix3 invMat) {
    final v = invMat.transform(vmath.Vector3(pt.dx, pt.dy, 1));
    return Offset(v.x, v.y);
  }

  List<Offset> mapKeypointsToOriginal(List<Keypoint> keypoints, vmath.Matrix3 invMat, int patchSize) {
    // Python logic: scale from output shape to input shape, then apply inverse affine
    List<Offset> mapped = [];
    for (var kp in keypoints) {
      // Scale from model output (e.g., 32x32) to input (256x256)
      double xScaled = kp.x / outputSize * inputSize;
      double yScaled = kp.y / outputSize * inputSize;
      Offset patchPt = Offset(xScaled, yScaled);
      mapped.add(applyInverseAffine(patchPt, invMat));
    }
    return mapped;
  }

  Future<void> processFrame(CameraImage cameraImage) async {
    img.Image? image = _convertYUV420ToImage(cameraImage);
    if (image == null) return;
    List<Rect> personBBoxes = await _detectPersons(image);
    for (Rect bbox in personBBoxes) {
      img.Image patch = _extractPersonPatch(image, bbox, inputSize);
      vmath.Matrix3 affine = computeAffineTransform(bbox, inputSize);
      OrtValue inputTensor = _prepareInputTensor(patch);
      final outputs = await _session.runAsync(OrtRunOptions(), {'input': inputTensor});
      if (outputs == null || outputs.isEmpty || outputs[0] == null) continue;
      final List outputData = outputs[0]!.value as List;
      final Float32List outputDataFloat = Float32List.fromList(outputData.cast<double>());
      List<Keypoint> keypoints = _parseKeypointsFromOutput(outputDataFloat);
      vmath.Matrix3 invAffine = invertAffineTransform(affine);
      // Use articulationVector as needed (draw, send, etc.)
      inputTensor.release();
      for (var ortValue in outputs) {
        ortValue?.release();
      }
    }
  }

  void _switchCamera() async {
    setState(() {
      _isCameraInitialized = false;
      _currentLensDirection =
        _currentLensDirection == CameraLensDirection.back
          ? CameraLensDirection.front
          : CameraLensDirection.back;
    });
    await _cameraController.dispose();
    await _initializeCamera();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    if (_isModelLoaded) {
      _session.release();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized) {
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Initializing Camera...'),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Pose Estimation with ONNX'),
        backgroundColor: Colors.blue,
        actions: [
          IconButton(
            icon: Icon(Icons.cameraswitch),
            tooltip: 'Switch Camera',
            onPressed: _switchCamera,
          ),
        ],
      ),
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          CustomPaint(
            painter: PosePainter(_allArticulationVectors, _latestYoloBBoxes),
            size: Size.infinite,
          ),
          if (!_isModelLoaded)
            const Positioned(
              top: 100,
              left: 0,
              right: 0,
              child: Center(
                child: Card(
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 8),
                        Text('Loading Model...'),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Center(
              child: Text(
                'Skeletons: \\${_allArticulationVectors.length}',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  shadows: [
                    Shadow(
                      blurRadius: 10.0,
                      color: Colors.black,
                      offset: Offset(2.0, 2.0),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // Ensure _detectPersons is defined as a method in _PoseEstimationPageState
  Future<List<Rect>> _detectPersons(img.Image image) async {
    // Placeholder for person detection logic
    // This should return a list of bounding boxes (Rect) for detected persons
    return [];
  }
}

class Keypoint {
  final double x;
  final double y;
  final double confidence;
  Keypoint({required this.x, required this.y, required this.confidence});
}

// LetterboxResult and letterbox function already at top-level
class LetterboxResult {
  final img.Image image;
  final double scale;
  final int padX;
  final int padY;
  LetterboxResult(this.image, this.scale, this.padX, this.padY);
}

LetterboxResult letterbox(img.Image src, int dstW, int dstH) {
  final srcW = src.width;
  final srcH = src.height;
  final scale = math.min(dstW / srcW, dstH / srcH);
  final newW = (srcW * scale).round();
  final newH = (srcH * scale).round();
  final resized = img.copyResize(src, width: newW, height: newH);
  final out = img.Image(width: dstW, height: dstH);
  img.fill(out, color: img.ColorRgb8(0, 0, 0));
  final padX = ((dstW - newW) / 2).round();
  final padY = ((dstH - newH) / 2).round();
  // Manual pixel copy fallback for compositing
  for (int y = 0; y < newH; y++) {
    for (int x = 0; x < newW; x++) {
      out.setPixel(padX + x, padY + y, resized.getPixel(x, y));
    }
  }
  return LetterboxResult(out, scale, padX, padY);
}

// Update PosePainter to accept and draw multiple skeletons
class PosePainter extends CustomPainter {
  final List<List<Offset>> allArticulationVectors;
  final List<Rect> yoloBBoxes;
  PosePainter(this.allArticulationVectors, [this.yoloBBoxes = const []]);

  @override
  void paint(Canvas canvas, Size size) {
    // Draw a test rectangle to confirm overlay alignment
    final testRectPaint = Paint()
      ..color = Colors.orange.withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), testRectPaint);
    // Draw a fixed test circle at the center
    final testCirclePaint = Paint()
      ..color = Colors.purple.withOpacity(0.7)
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(size.width / 2, size.height / 2), 30, testCirclePaint);
    // Print debug info
    print('Painter preview size: $size, articulationVectors count: ${allArticulationVectors.length}');

    final jointPaint = Paint()
      ..color = Colors.green
      ..strokeWidth = 6.0
      ..style = PaintingStyle.fill;

    final skeletonPaint = Paint()
      ..color = Colors.blueAccent
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;

    final bboxPaint = Paint()
      ..color = Colors.red.withOpacity(0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final skeleton = [
      [10, 9], [9, 8], [8, 11], [8, 14],
      [11, 12], [12, 13], [14, 15], [15, 16],
      [11, 4], [14, 1], [0, 4], [0, 1],
      [4, 5], [5, 6], [1, 2], [2, 3]
    ];

    for (final articulationVector in allArticulationVectors) {
      // If values are normalized (0-1), scale to preview size
      bool isNormalized = articulationVector.any((pt) => pt.dx > 0 && pt.dx < 1 && pt.dy > 0 && pt.dy < 1);
      List<Offset> pts = isNormalized
        ? articulationVector.map((pt) => Offset(pt.dx * size.width, pt.dy * size.height)).toList()
        : articulationVector;
      // Draw joints
      for (final pt in pts) {
        if (pt.dx >= 0 && pt.dx < size.width && pt.dy >= 0 && pt.dy < size.height) {
          canvas.drawCircle(pt, 4, jointPaint);
        }
      }
      // Draw skeleton
      for (final pair in skeleton) {
        if (pair[0] < pts.length && pair[1] < pts.length) {
          final p1 = pts[pair[0]];
          final p2 = pts[pair[1]];
          if (
            p1.dx >= 0 && p1.dx < size.width && p1.dy >= 0 && p1.dy < size.height &&
            p2.dx >= 0 && p2.dx < size.width && p2.dy >= 0 && p2.dy < size.height
          ) {
            canvas.drawLine(p1, p2, skeletonPaint);
          }
        }
      }
    }
    // Draw YOLO bounding boxes in red
    for (final bbox in yoloBBoxes) {
      canvas.drawRect(bbox, bboxPaint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}