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

"""Helper function to decode ctc trained model's output."""


def decode_ctc_conf(pred,
                    classes,
                    blank_id):
    """Decode ctc trained model's output.

    Return decoded license plate and confidence.
    """
    pred_id = pred[0]
    pred_conf = pred[1]
    decoded_lp = []
    decoded_conf = []

    for idx_in_batch, seq in enumerate(pred_id):
        seq_conf = pred_conf[idx_in_batch]
        prev = seq[0]
        tmp_seq = [prev]
        tmp_conf = [seq_conf[0]]
        for idx in range(1, len(seq)):
            if seq[idx] != prev:
                tmp_seq.append(seq[idx])
                tmp_conf.append(seq_conf[idx])
                prev = seq[idx]
        lp = ""
        output_conf = []
        for index, i in enumerate(tmp_seq):
            if i != blank_id:
                lp += classes[i]
                output_conf.append(tmp_conf[index])
        decoded_lp.append(lp)
        decoded_conf.append(output_conf)

    return decoded_lp, decoded_conf
