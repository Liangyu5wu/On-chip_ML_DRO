import hls4ml
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model



# model = load_model("QAT_32_16_8_model.h5")
model = load_model("32_16_8_model.h5")

config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
print("-----------------------------------")
print("Configuration")
print(config)
print("-----------------------------------")
config['Model']['Strategy']= 'Latency'
hls_model = hls4ml.converters.create_config(backend='Vitis')
hls_model['HLSConfig'] = config
hls_model['KerasModel'] = model
hls_model['ClockPeriod'] = 25
hls_model['OutputDir'] = 'TEST_HLS'
hls_model['IOType'] = 'io_parallel'
hls_model['Part'] = 'xcvu9p-flga2104-2-e'
hls_model_compile = hls4ml.converters.keras_to_hls(hls_model)
hls_model_compile.compile()
# hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True)
# hls4ml.utils.fetch_example_list()
hls_model_compile.build(csim=False)
# hls4ml.report.read_vivado_report('my-hls-test')
