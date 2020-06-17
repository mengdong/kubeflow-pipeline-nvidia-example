# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kfp.dsl as dsl
import kfp.gcp as gcp
from kubernetes import client as k8s_client

@dsl.pipeline(
  name='End2end Resnet50 Classification',
  description='END to END kubeflow demo with TensorRT Inference Server, TF-AMP and TensorRT'
)
def end2end_demo(  # pylint: disable=unused-argument
    trtserver_name: dsl.PipelineParam = dsl.PipelineParam(name='trtserver_name', value='trtserver'),
    model_name: dsl.PipelineParam = dsl.PipelineParam(name='model_name', value='resnet_graphdef'),
    model_version: dsl.PipelineParam = dsl.PipelineParam(name='model_version', value='1'),
    num_iter: dsl.PipelineParam = dsl.PipelineParam(name='num_iter', value=10),
    batch_size: dsl.PipelineParam = dsl.PipelineParam(name='batch_size', value=128),
    webapp_prefix: dsl.PipelineParam = dsl.PipelineParam(name='webapp_prefix', value='webapp'),
    webapp_port: dsl.PipelineParam = dsl.PipelineParam(name='webapp_port', value='80'),
    storage_bucket: dsl.PipelineParam = dsl.PipelineParam(name='storage_bucket', value='gs://dongm-kubeflow'),
    ckpt_dir: dsl.PipelineParam = dsl.PipelineParam(name='ckpt_dir', value='ckpt'),
    mount_dir: dsl.PipelineParam = dsl.PipelineParam(name='mount_dir', value='/mnt/vol'),
    model_dir: dsl.PipelineParam = dsl.PipelineParam(name='model_dir', value='test_model_repository'),
    raw_data_dir: dsl.PipelineParam = dsl.PipelineParam(name='raw_data_dir', value='raw_data'),
    processed_data_dir: dsl.PipelineParam = dsl.PipelineParam(name='processed_data_dir', value='processed_data')
):
  project_name = 'nvidia-sa-org'

  preprocessing = dsl.ContainerOp(
    name='preprocessing',
    image='gcr.io/' + project_name + '/gcp-end2end-demo-preprocessing',
    command=['python'],
    arguments=[
      '/scripts/preprocess.py',
      '--input_dir', '%s/%s' % (mount_dir, raw_data_dir),
      '--output_dir', '%s/%s' % (mount_dir, processed_data_dir)
    ],
    file_outputs={}
  ).add_volume(k8s_client.V1Volume(name='my-rw-pv',
                                   persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                                     claim_name='my-rw-pvc'))
               ).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/mnt/vol', name='my-rw-pv')).set_gpu_limit(1)

  training = dsl.ContainerOp(
    name='training',
    image='gcr.io/' + project_name + '/gcp-end2end-demo-training',
    command=['python'],
    arguments=[
      '/scripts/train.py',
      '--model_version', str(model_version),
      '--input_dir', '%s/%s' % (mount_dir, processed_data_dir),
      '--ckpt_dir', '%s/%s' % (mount_dir, ckpt_dir),
      '--output_dir', '%s/%s/resnet_graphdef' % (storage_bucket, model_dir),
      '--num_iter', '%s' % (num_iter),
      '--batch_size', '%s' % (batch_size)
    ],
    file_outputs={}
  ).set_gpu_limit(1).add_volume(k8s_client.V1Volume(name='my-rw-pv',
                                                    persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                                                      claim_name='my-rwm-pvc'))
                                ).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/mnt/vol', name='my-rw-pv')).apply(gcp.use_gcp_secret('user-nvidia-sa'))

  serve = dsl.ContainerOp(
    name='serve',
    image='gcr.io/' + project_name + '/ml-pipeline-kubeflow-trtisserve',
    arguments=["--trtserver_name", trtserver_name,
               '--model_version', model_version,
               '--orig_model_path', '%s/%s' % (mount_dir, model_name),
               "--model_path", '%s/%s' % (storage_bucket, model_dir)
               ]
  )

  webapp = dsl.ContainerOp(
    name='webapp',
    image='gcr.io/' + project_name + '/ml-pipeline-trtis-webapp-launcher',
    arguments=["--workflow_name", '%s' % ('{{workflow.name}}',),
               "--trtserver_name", trtserver_name,
               "--model_name", model_name,
               "--model_version", str(model_version),
               "--webapp_prefix", webapp_prefix,
               "--webapp_port", str(webapp_port)
               ]
  )

  training.after(preprocessing)
  serve.after(training)
  webapp.after(serve)


if __name__ == '__main__':
  import kfp.compiler as compiler

  compiler.Compiler().compile(end2end_demo, __file__ + '.tar.gz')
