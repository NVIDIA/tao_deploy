# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference module."""
import numpy as np
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from nvidia_tao_deploy.cv.classification_tf1.inferencer import ClassificationInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference


def trt_output_process_fn(out):
    """Function to process TRT model output."""
    return np.reshape(out.host, out.numpy_shape)


class MLRecogInferencer(ClassificationInferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, engine_path, input_shape=None, batch_size=None):
        """Initializes TensorRT objects needed for model inference.

        Args:
            engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
        """
        # Load TRT engine
        super().__init__(engine_path, input_shape, batch_size, "channel_first")

    def train_knn(self, gallery_dl, k=1):
        """Trains KNN model on gallery dataset.

        Args:
            gallery_dl (MLRecogClassificationLoader): gallery dataset loader
            k (int): number of nearest neighbors to use
        """
        image_embeds = []
        labels = []
        for imgs, labs in tqdm(gallery_dl):
            image_embeds.append(self.get_embeddings(imgs))
            labels.append(labs)
        image_embeds = np.concatenate(image_embeds)
        labels = np.concatenate(labels)
        self.knn = KNeighborsClassifier(n_neighbors=k, metric="l2")
        self.knn.fit(image_embeds, labels)

    def infer(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            query (numpy.ndarray): batch of image numpy arrays

        Returns:
            dists (numpy.ndarray): distances to the nearest neighbors
            indices (numpy.ndarray): indices of the nearest neighbors
        """
        query_emb = self.get_embeddings(imgs)
        dists, indices = self.knn.kneighbors(query_emb)
        return dists, indices

    def get_embeddings(self, imgs):
        """Returns normalized embeddings for a batch of images.

        Args:
            imgs (np.ndarray): batch of numpy arrays of images

        Returns:
            x_emb (np.ndarray): normalized embeddings
        """
        x_emb = self.get_embeddings_from_batch(imgs)
        x_emb = normalize(x_emb, norm="l2", axis=1)
        return x_emb

    def get_embeddings_from_batch(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            imgs (np.ndarray): batch of numpy arrays of images

        Returns:
            x_emb (np.ndarray): embeddings output from trt engine
        """
        # Wrapped in list since arg is list of named tensor inputs
        # For metric learning, there is just 1: [input]
        self._copy_input_to_host([imgs])

        # ...fetch model outputs...
        # 1 named result: [fc_pred]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Process TRT outputs to proper format
        return trt_output_process_fn(results[0])
