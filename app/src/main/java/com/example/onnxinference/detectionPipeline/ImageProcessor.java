package com.example.onnxinference.detectionPipeline;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import com.example.onnxinference.detectionPipeline.Detection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ImageProcessor {

    public float[][][][] preprocessImage(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true);
        float[][][][] input = new float[1][3][640][640];
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                input[0][0][y][x] = (pixel >> 16 & 0xFF) / 255.0f; // R
                input[0][1][y][x] = (pixel >> 8 & 0xFF) / 255.0f;  // G
                input[0][2][y][x] = (pixel & 0xFF) / 255.0f;       // B
            }
        }
        return input;
    }

    public Bitmap processOutput(float[][][] output, Bitmap originalBitmap, List<String> classNames, float confidenceThreshold, float iouThreshold) {
        int numDetections = 8400;
        int numClasses = 80;
        int probabilityStartIndex = 4;

        int originalWidth = originalBitmap.getWidth();
        int originalHeight = originalBitmap.getHeight();
        float scaleX = originalWidth / 640.0f;
        float scaleY = originalHeight / 640.0f;

        List<Detection> detections = new ArrayList<>();

        for (int i = 0; i < numDetections; i++) {
            float centerX = output[0][0][i] * scaleX;
            float centerY = output[0][1][i] * scaleY;
            float width = output[0][2][i] * scaleX;
            float height = output[0][3][i] * scaleY;

            int classId = -1;
            float maxClassProb = 0;
            for (int j = probabilityStartIndex; j < probabilityStartIndex + numClasses; j++) {
                float classProb = output[0][j][i];
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = j - probabilityStartIndex;
                }
            }

            if (maxClassProb > confidenceThreshold) {
                detections.add(new Detection(centerX - width / 2, centerY - height / 2, width, height, maxClassProb, classId));
            }
        }

        List<Detection> nmsDetections = applyNMS(detections, iouThreshold);

        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);
        Paint textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(20);

        for (Detection detection : nmsDetections) {
            canvas.drawRect(detection.x, detection.y, detection.x + detection.width, detection.y + detection.height, paint);
            String className = classNames.get(detection.classId);
            canvas.drawText(className + ": " + String.format("%.2f", detection.confidence), detection.x, detection.y - 10, textPaint);
        }

        return mutableBitmap;
    }

    private List<Detection> applyNMS(List<Detection> detections, float iouThreshold) {
        List<Detection> nmsDetections = new ArrayList<>();
        Collections.sort(detections, (d1, d2) -> Float.compare(d2.confidence, d1.confidence));

        while (!detections.isEmpty()) {
            Detection bestDetection = detections.remove(0);
            nmsDetections.add(bestDetection);

            detections.removeIf(detection -> calculateIoU(bestDetection, detection) > iouThreshold);
        }

        return nmsDetections;
    }

    private float calculateIoU(Detection d1, Detection d2) {
        RectF rect1 = d1.toRectF();
        RectF rect2 = d2.toRectF();

        float intersectionLeft = Math.max(rect1.left, rect2.left);
        float intersectionTop = Math.max(rect1.top, rect2.top);
        float intersectionRight = Math.min(rect1.right, rect2.right);
        float intersectionBottom = Math.min(rect1.bottom, rect2.bottom);

        if (intersectionLeft < intersectionRight && intersectionTop < intersectionBottom) {
            float intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
            float unionArea = d1.getArea() + d2.getArea() - intersectionArea;
            return intersectionArea / unionArea;
        }

        return 0;
    }
}
