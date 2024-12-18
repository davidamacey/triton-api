from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")  # load an official model

# Retreive metadata during export
metadata = []

def export_cb(exporter):
    metadata.append(exporter.metadata)


model.add_callback("on_export_end", export_cb)

# Export the model
onnx_file = model.export(format="onnx", dynamic=True, device = "cuda:1")

# (Optional) Enable TensorRT for GPU inference
# First run will be slow due to TensorRT engine conversion
data = """
optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      name: "tensorrt"
      parameters {
        key: "precision_mode"
        value: "FP16"
      }
      parameters {
        key: "max_workspace_size_bytes"
        value: "3221225472"
      }
      parameters {
        key: "trt_engine_cache_enable"
        value: "1"
      }
      parameters {
        key: "trt_engine_cache_path"
        value: "/models/yolo/1"
      }
    }
  }
}
parameters {
  key: "metadata"
  value: {
    string_value: "%s"
  }
}
""" % metadata[0]

with open("generated_config.pbtxt", "w") as f:
    f.write(data)