import tf.transformations as tf_trans
def p2T(pose):
    translation = tf_trans.translation_matrix(pose[:3])
    rotation = tf_trans.euler_matrix(pose[3], pose[4], pose[5])
    transformation_matrix = tf_trans.concatenate_matrices(translation, rotation)
    return transformation_matrix