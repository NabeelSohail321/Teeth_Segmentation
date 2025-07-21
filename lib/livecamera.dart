import 'dart:io';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart' hide YOLOResult;
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/src/painting/decoration.dart ' hide BoxPainter;

import 'main.dart'; // if YOLOResult and BoxPainter are defined in main.dart

class CameraViewPage extends StatefulWidget {
  @override
  _CameraViewPageState createState() => _CameraViewPageState();
}

class _CameraViewPageState extends State<CameraViewPage> {
  YOLOViewController? _yoloController;
  List<YOLOResult> _results = [];
  Size _cameraSize = Size.zero;

  Future<String> _copyModelToTemp() async {
    final data = await rootBundle.load('assets/best_float32.tflite');
    final bytes = data.buffer.asUint8List();
    final tmpDir = await getTemporaryDirectory();
    final modelFile = File('${tmpDir.path}/best_float32.tflite');
    await modelFile.writeAsBytes(bytes, flush: true);
    return modelFile.path;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        automaticallyImplyLeading: false,
        title: Text('Live Camera Detection'),
        actions: [
          IconButton(onPressed: (){
            Navigator.pop(context);
          }, icon: Icon(Icons.arrow_back)),
          IconButton(
            icon: Icon(Icons.cameraswitch),
            onPressed: () async {
              try {
                await _yoloController?.switchCamera();
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Unable to switch camera')),
                );
              }
            },
          ),
        ],
      ),
      body: FutureBuilder<String>(
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

              return Stack(
                children: [
                  YOLOView(
                    controller: _yoloController,
                    modelPath: modelPath,
                    task: YOLOTask.segment,
                    onResult: (results) {
                      if (results is List) {
                        final newResults = results
                            .where((r) => r != null)
                            .map((r) => YOLOResult.fromMap(r as Map<String, dynamic>))
                            .toList();
                        setState(() => _results = newResults);
                      }
                    },
                  ),
                  RepaintBoundary(
                    child: CustomPaint(
                      painter: TeethBoxPainter(_results, _cameraSize),
                      child: Container(),
                    ),
                  ),
                ],
              );
            },
          );
        },
      ),
    );
  }
}
