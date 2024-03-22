import tensorflow as tf

# Load the .pb file
with tf.io.gfile.GFile('trip_updates.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Load the graph def into a new graph
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

# Now you can use the operations in the graph
# For example, if you have an input placeholder named 'input_tensor'
# and an output tensor named 'output_tensor', you can run inference like this:
#with tf.compat.v1.Session(graph=graph) as sess:
   # input_tensor = graph.get_tensor_by_name('input_tensor_name:0')  # Adjust name and index as needed
   # output_tensor = graph.get_tensor_by_name('output_tensor_name:0')  # Adjust name and index as needed

    # Run inference
   # output = sess.run(output_tensor, feed_dict={input_tensor: your_input_data})