from ann_visualizer.visualize import ann_viz
import tensorflow as tf
from ProGANVanilla import *
from datetime import datetime

#@tf.function
#def vis():

    # input_shape = (16, 4, 4, 3)
    # x = tf.random.normal(input_shape)

    # model(x, training=True)
  #  ann_viz(model, title="My first neural network")

#vis()

model = Prog_Discriminator()
@tf.function
def feed(x):
    return model(x)

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

input_shape = (1, 4, 4, 3)
x = tf.random.normal(input_shape)
# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
feed(x)
with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)