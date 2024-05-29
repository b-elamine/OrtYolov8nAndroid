package com.example.onnxinference;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
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
    private TextView text;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classNames = loadClassNames();

        Yolov8Inference yolo = new Yolov8Inference(this);

        text = (TextView) findViewById(R.id.text);


        Bitmap bitmap = loadImageFromAssets("test.jpg");
        float[][][][] inputTensor = preprocessImage(bitmap);

        float[][][] result = yolo.runInference(inputTensor);

        processOutput(result);
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

    private void processOutput(float[][][] output) {
        // Confidence threshold to filter out low-confidence detections
        float confidenceThreshold = 0.5f;

        int numDetections = 8400; // Number of detections
        int numClasses = 80; // Number of classes (0 to 79)
        int probabilityStartIndex = 4; // Index where class probabilities start
        int count = 0;

        // Iterate over each bounding box prediction
        for (int i = 0; i < numDetections; i++) {
            // Extract bounding box coordinates
            float centerX = output[0][0][i];
            float centerY = output[0][1][i];
            float width = output[0][2][i];
            float height = output[0][3][i];

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
                count++;
                // Print bounding box, class name, and confidence
                if (classId != -1 && classId < classNames.size()) {
                    String className = classNames.get(classId);
                    String txt = String.format("Detected %s with confidence %.2f at [x=%.2f, y=%.2f, w=%.2f, h=%.2f]",
                            className, maxClassProb, centerX, centerY, width, height);
                    System.out.println(txt);
                    text.append(txt + "\n"); // Append to TextView to display multiple detections
                }
            }
        }
    }}
