import numpy as np
import cv2
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, PReLU, Add, Input


def srresnet(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])
        return m

    inputs = Input(shape=(None, None, input_channel_num))
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model


def demo(vis_type):
    model = srresnet()
    model.load_weights('demo.hdf5')

    cap = cv2.VideoCapture('demo.avi')
    total_frame_num_VIDEO = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if vis_type == 'write':
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        dstSize = (int(frameWidth), int(frameHeight))
        out = cv2.VideoWriter("demo_denoised.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, dstSize)

    for frame_idx in range(total_frame_num_VIDEO):
        ret, frame = cap.read()

        pred = model.predict(np.expand_dims(frame, 0))
        inference = np.clip(pred[0], 0, 255).astype(dtype=np.uint8)

        if vis_type == 'write':
            out.write(inference)
        else:
            cv2.imshow('inference', inference)
            cv2.waitKey(1)


if __name__ == "__main__":
    vis = ['write', 'show']
    demo(vis[0])
