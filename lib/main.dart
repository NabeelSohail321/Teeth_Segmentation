import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: SegmentationPage());
  }
}

class SegmentationPage extends StatefulWidget {
  @override
  _SegmentationPageState createState() => _SegmentationPageState();
}

class _SegmentationPageState extends State<SegmentationPage> {
  List<YOLOResult> _results = [];
  Uint8List? _imageBytes;
  YOLO? _yolo;
  final YOLOViewController _yoloController = YOLOViewController();
  bool _useCamera = false;
  bool _isLoading = false;
  Size _cameraSize = Size.zero;
  bool showYoloView = false;

  Future<String> _copyModelToTemp() async {
    final data = await rootBundle.load('assets/best_float32.tflite');
    final bytes = data.buffer.asUint8List();
    final tmpDir = await getTemporaryDirectory();
    final modelFile = File('${tmpDir.path}/best_float32.tflite');
    await modelFile.writeAsBytes(bytes, flush: true);
    return modelFile.path;
  }

  Future<void> _runSegmentationOnImage(XFile file) async {
    try {
      final bytes = await file.readAsBytes();
      setState(() {
        _isLoading = true;
        _imageBytes = bytes;
        _results = [];
        _useCamera = false;
      });

      final modelPath = await _copyModelToTemp();

      _yolo ??= YOLO(
        modelPath: modelPath,
        task: YOLOTask.segment,
      );
      await _yolo!.loadModel();

      final resultMap = await _yolo!.predict(bytes);

      if (resultMap.containsKey('predictions')) {
        final preds = (resultMap['predictions'] as List?) ?? [];
        setState(() {
          _results = preds
              .where((p) => p != null)
              .map((p) => YOLOResult.fromMap(p as Map<String, dynamic>))
              .toList();
        });
      } else if (resultMap.containsKey('boxes')) {
        final boxes = (resultMap['boxes'] as List?) ?? [];
        setState(() {
          _results = boxes
              .where((b) => b != null)
              .map((b) => YOLOResult(
            box: Rect.fromLTRB(
              b['x1']?.toDouble() ?? 0,
              b['y1']?.toDouble() ?? 0,
              b['x2']?.toDouble() ?? 0,
              b['y2']?.toDouble() ?? 0,
            ),
            confidence: b['confidence']?.toDouble() ?? 0,
            classIndex: 0,
            className: b['class']?.toString() ?? 'unknown',
          ))
              .toList();
        });
      }
    } catch (e, stack) {
      print('Error during image segmentation: $e\n$stack');
      _showErrorSnackbar('Image processing failed.');
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _showErrorSnackbar(String message) {
    if (context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
    }
  }

  Widget _buildChoiceScreen() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          ElevatedButton.icon(
            icon: Icon(Icons.photo),
            label: Text('Pick Image from Gallery'),
            onPressed: () async {
              final picker = ImagePicker();
              final file = await picker.pickImage(source: ImageSource.gallery);
              if (file != null) await _runSegmentationOnImage(file);
            },
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            icon: Icon(Icons.camera_alt),
            label: Text('Use Camera'),
            onPressed: () {
              setState(() {

                _useCamera = true;
                _imageBytes = null;
                _results = [];
                showYoloView = true;
              });
            },
          ),
        ],
      ),
    );
  }

  Widget _buildCameraView() {
    return FutureBuilder<String>(
      future: _copyModelToTemp(),
      builder: (context, snapshot) {
        if (!snapshot.hasData) {
          return Center(child: CircularProgressIndicator());
        }

        final modelPath = snapshot.data!;

        return LayoutBuilder(
          builder: (context, constraints) {
            if (_cameraSize == Size.zero) {
              _cameraSize = Size(constraints.maxWidth, constraints.maxHeight);
            }

            if(showYoloView){
              return Stack(
                children: [
                  Container(
                    color: Colors.black,
                    child: YOLOView(
                      controller: _yoloController,
                      modelPath: modelPath,
                      task: YOLOTask.segment,


                      onResult: (results) {
                        if (results is List) {
                          final List<YOLOResult> newResults = results
                              .where((r) => r != null)
                              .map((r) => YOLOResult.fromMap(r as Map<String, dynamic>))
                              .toList();

                          setState(() {
                            _results = newResults;
                          });
                        }
                      },
                    ),
                  ),
                  RepaintBoundary(
                    child: CustomPaint(
                      painter: TeethBoxPainter(_results, _cameraSize),
                      child: Container(),
                    ),
                  ),
                ],
              );
            }else{
              return Container();
            }
          },
        );
      },
    );
  }

  Widget _buildImageView() {
    if (_imageBytes == null) return SizedBox();

    return SingleChildScrollView(
      child: Column(
        children: [
          Stack(
            alignment: Alignment.center,
            children: [
              FutureBuilder<ImageInfo>(
                future: getImageInfo(_imageBytes!),
                builder: (context, snapshot) {
                  if (!snapshot.hasData) return CircularProgressIndicator();

                  final imageInfo = snapshot.data!;
                  final imageSize = Size(
                    imageInfo.image.width.toDouble(),
                    imageInfo.image.height.toDouble(),
                  );

                  return LayoutBuilder(
                    builder: (context, constraints) {
                      return Stack(
                        children: [
                          Image.memory(_imageBytes!,
                              fit: BoxFit.contain, width: constraints.maxWidth),
                          CustomPaint(
                            size: Size(constraints.maxWidth,
                                constraints.maxWidth * imageSize.height / imageSize.width),
                            painter: TeethBoxPainter(_results, imageSize),
                          ),
                        ],
                      );
                    },
                  );
                },
              ),
              if (_isLoading)
                Container(
                  color: Colors.black.withOpacity(0.3),
                  child: const Center(child: CircularProgressIndicator()),
                ),
            ],
          ),
          const SizedBox(height: 16),
          if (_results.isNotEmpty)
            ListView.builder(
              physics: NeverScrollableScrollPhysics(),
              shrinkWrap: true,
              itemCount: _results.length,
              itemBuilder: (context, index) {
                final result = _results[index];
                return Card(
                  elevation: 4,
                  margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Class: ${result.className}',
                            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                        Text('Confidence: ${(result.confidence * 100).toStringAsFixed(2)}%',
                            style: TextStyle(fontSize: 14)),
                        Text(
                            'Box: [${result.box.left.toStringAsFixed(1)}, ${result.box.top.toStringAsFixed(1)}] → [${result.box.right.toStringAsFixed(1)}, ${result.box.bottom.toStringAsFixed(1)}]',
                            style: TextStyle(fontSize: 14)),
                      ],
                    ),
                  ),
                );
              },
            ),
        ],
      ),
    );
  }

  Future<ImageInfo> getImageInfo(Uint8List bytes) async {
    final Completer<ImageInfo> completer = Completer();
    final img = Image.memory(bytes);
    img.image.resolve(ImageConfiguration()).addListener(
      ImageStreamListener((ImageInfo info, bool _) {
        completer.complete(info);
      }),
    );
    return completer.future;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Teeth Segmentation'),
        actions: [
          IconButton(
            icon: Icon(Icons.refresh),
            onPressed: () {
              // Dispose old controller if exists
              setState(() {
                showYoloView=false;
                _imageBytes = null;
                _results = [];
                _useCamera = false;
              });
            },
          ),
          if (_useCamera)
            IconButton(
              icon: Icon(Icons.cameraswitch),
              onPressed: () async {
                try {
                  await _yoloController?.switchCamera();
                } catch (e) {
                  _showErrorSnackbar('Unable to switch camera.');
                }
              },
            ),
        ],
      ),
      body: _imageBytes != null
          ? _buildImageView()
          : (_useCamera ? _buildCameraView() : _buildChoiceScreen()),
    );
  }
}

class YOLOResult {
  final Rect box;
  final double confidence;
  final int classIndex;
  final String className;
  final List<List<double>>? mask;

  YOLOResult({
    required this.box,
    required this.confidence,
    required this.classIndex,
    required this.className,
    this.mask,
  });

  factory YOLOResult.fromMap(Map<String, dynamic> map) {
    return YOLOResult(
      box: Rect.fromLTRB(
        (map['x1'] as num?)?.toDouble() ?? 0,
        (map['y1'] as num?)?.toDouble() ?? 0,
        (map['x2'] as num?)?.toDouble() ?? 0,
        (map['y2'] as num?)?.toDouble() ?? 0,
      ),
      confidence: (map['confidence'] as num?)?.toDouble() ?? 0,
      classIndex: (map['classIndex'] as int?) ?? 0,
      className: (map['className'] as String?) ?? 'Unknown',
      mask: (map['mask'] != null)
          ? List<List<double>>.from(
          (map['mask'] as List).map((row) => List<double>.from(row)))
          : null,
    );
  }
}

class TeethBoxPainter extends CustomPainter {
  final List<YOLOResult> results;
  final Size imageSize;

  TeethBoxPainter(this.results, this.imageSize);

  @override
  void paint(Canvas canvas, Size size) {
    if (results.isEmpty) return;

    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    final paintBox = Paint()
      ..style = PaintingStyle.stroke
      ..color = Colors.red
      ..strokeWidth = 2;

    final paintMask = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.red.withOpacity(0.3);

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    for (var r in results) {
      final left = r.box.left * scaleX;
      final top = r.box.top * scaleY;
      final right = r.box.right * scaleX;
      final bottom = r.box.bottom * scaleY;

      final rect = Rect.fromLTRB(left, top, right, bottom);
      canvas.drawRect(rect, paintBox);

      if (r.mask != null) {
        final path = Path();
        // Mask data assumed to be relative coordinates [ [x,y], ... ]
        for (var i = 0; i < r.mask!.length; i++) {
          final point = r.mask![i];
          final px = point[0] * scaleX;
          final py = point[1] * scaleY;
          if (i == 0) {
            path.moveTo(px, py);
          } else {
            path.lineTo(px, py);
          }
        }
        path.close();
        canvas.drawPath(path, paintMask);
      }

      final label = '${r.className} ${(r.confidence * 100).toStringAsFixed(1)}%';
      final textSpan = TextSpan(
        text: label,
        style: TextStyle(
          color: Colors.white,
          fontSize: 14,
          backgroundColor: Colors.black54,
        ),
      );
      textPainter.text = textSpan;
      textPainter.layout();
      textPainter.paint(canvas, Offset(left, top - textPainter.height));
    }
  }

  @override
  bool shouldRepaint(covariant TeethBoxPainter oldDelegate) {
    return oldDelegate.results != results || oldDelegate.imageSize != imageSize;
  }
}
