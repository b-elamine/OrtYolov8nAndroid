package com.example.onnxinference;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private List<String> classNames;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classNames = loadClassNames();

        Yolov8Inference yolo = new Yolov8Inference(this);

        imageView = findViewById(R.id.imageView);

        Bitmap bitmap = loadImageFromAssets("test.jpg");
        float[][][][] inputTensor = preprocessImage(bitmap);

        float[][][] result = yolo.runInference(inputTensor);

        Bitmap resultBitmap = processOutput(result, bitmap);
        imageView.setImageBitmap(resultBitmap);
    }

    private List<String> loadClassNames() {
        List<String> classNames = new ArrayList<>();
        try {
            InputStream inputStream = getResources().openRawResource(R.raw.labels);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = reader.readLine()) != null) {
                classNames.add(line.trim());
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private Bitmap loadImageFromAssets(String fileName) {
        try {
            InputStream inputStream = getAssets().open(fileName);
            return BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private float[][][][] preprocessImage(Bitmap bitmap) {
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

    private Bitmap processOutput(float[][][] output, Bitmap originalBitmap) {
        // Confidence threshold to filter out low-confidence detections
        float confidenceThreshold = 0.5f;

        int numDetections = 8400; // Number of detections
        int numClasses = 80; // Number of classes (0 to 79)
        int probabilityStartIndex = 4; // Index where class probabilities start

        int originalWidth = originalBitmap.getWidth();
        int originalHeight = originalBitmap.getHeight();
        float scaleX = originalWidth / 640.0f;
        float scaleY = originalHeight / 640.0f;

        // Create a mutable copy of the original bitmap
        Bitmap mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);
        Paint textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(20);

        for (int i = 0; i < numDetections; i++) {
            // Extract bounding box coordinates
            float centerX = output[0][0][i] * scaleX;
            float centerY = output[0][1][i] * scaleY;
            float width = output[0][2][i] * scaleX;
            float height = output[0][3][i] * scaleY;

            // Find the class with the highest probability for every detected box
            int classId = -1;
            float maxClassProb = 0;
            for (int j = probabilityStartIndex; j < probabilityStartIndex + numClasses; j++) {
                float classProb = output[0][j][i];
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = j - probabilityStartIndex;
                }
            }

            // Only consider detections with confidence above the threshold
            if (maxClassProb > confidenceThreshold) {
                // Print bounding box, class name, and confidence
                if (classId != -1 && classId < classNames.size()) {
                    String className = classNames.get(classId);
                    float x = centerX - width / 2;
                    float y = centerY - height / 2;

                    // Draw the bounding box
                    canvas.drawRect(x, y, x + width, y + height, paint);
                    // Draw the class name and confidence
                    canvas.drawText(className + ": " + String.format("%.2f", maxClassProb), x, y - 10, textPaint);

                    String result = String.format("Detected %s with confidence %.2f at [x=%.2f, y=%.2f, w=%.2f, h=%.2f]",
                            className, maxClassProb, centerX, centerY, width, height);
                    System.out.println(result);
                    //text.append(result + "\n"); // Append to TextView to display multiple detections
                }
            }
        }

        return mutableBitmap;
    }

}
