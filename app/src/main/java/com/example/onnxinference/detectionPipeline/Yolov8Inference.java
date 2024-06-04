package com.example.onnxinference.detectionPipeline;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.content.Context;

import com.example.onnxinference.R;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;

public class Yolov8Inference {
    private OrtEnvironment env;
    private OrtSession session;

    public Yolov8Inference(Context context) {
        try {
            env = OrtEnvironment.getEnvironment();
            File modelFile = getModelFile(context, "yolov8.onnx");
            session = env.createSession(modelFile.getPath());
        } catch (IOException | OrtException e) {
            e.printStackTrace();
        }
    }

    private File getModelFile(Context context, String modelName) throws IOException {
        InputStream inputStream = context.getResources().openRawResource(R.raw.yolov8n);
        File modelFile = new File(context.getFilesDir(), modelName);
        try (FileOutputStream outputStream = new FileOutputStream(modelFile)) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }
        return modelFile;
    }

    public float[][][] runInference(float[][][][] inputTensor) {
        try {
            OnnxTensor input = OnnxTensor.createTensor(env, inputTensor);
            System.out.println("RUNNING INFERENCE");
            OrtSession.Result result = session.run(Collections.singletonMap("images", input));
            System.out.println("DONE");
            float[][][] output = (float[][][]) result.get(0).getValue();
            return output;
        } catch (OrtException e) {
            e.printStackTrace();
            return null;
        }
    }
}
