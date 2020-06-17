# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import logging
import subprocess
import requests


def main():
  parser = argparse.ArgumentParser(description='Serving webapp')
  parser.add_argument(
      '--trtserver_name', default='trtserver',
      help='...',
      required=True)
  parser.add_argument(
      '--workflow_name', default='testworkflow',
      help='...',
      required=True)
  parser.add_argument(
      '--model_name', default='resnet_graphdef',
      help='...',
      required=True)
  parser.add_argument(
      '--model_version', default='1',
      help='...',
      required=True)
  parser.add_argument(
      '--webapp_prefix', default='webapp',
      help='...',
      required=True)
  parser.add_argument(
      '--webapp_port', default='80',
      help='...',
      required=True)
  parser.add_argument('--cluster', type=str,
                      help='GKE cluster set up for kubeflow. If set, zone must be provided. ' +
                           'If not set, assuming this runs in a GKE container and current ' +
                           'cluster is used.')
  parser.add_argument('--zone', type=str, help='zone of the kubeflow cluster.')
  args = parser.parse_args()

  KUBEFLOW_NAMESPACE = 'kubeflow'

  print("using model name: %s and namespace: %s" % (args.model_name, KUBEFLOW_NAMESPACE))

  logging.getLogger().setLevel(logging.INFO)
  args_dict = vars(args)

  if args.cluster and args.zone:
    cluster = args_dict.pop('cluster')  #pylint: disable=unused-variable
    zone = args_dict.pop('zone')  #pylint: disable=unused-variable
  else:
    # Get cluster name and zone from metadata
    metadata_server = "http://metadata/computeMetadata/v1/instance/"
    metadata_flavor = {'Metadata-Flavor' : 'Google'}
    cluster = requests.get(metadata_server + "attributes/cluster-name",
                           headers=metadata_flavor).text
    zone = requests.get(metadata_server + "zone",
                        headers=metadata_flavor).text.split('/')[-1]

  # logging.info('Getting credentials for GKE cluster %s.' % cluster)
  # subprocess.call(['gcloud', 'container', 'clusters', 'get-credentials', cluster,
  #                  '--zone', zone])

  logging.info('Generating training template.')

  template_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trtis-clientapp-template.yaml')
  target_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trtis-clientapp.yaml')

  with open(template_file, 'r') as f:
    with open(target_file, "w") as target:
      data = f.read()
      changed = data.replace('MODEL_PASSIN_NAME', args.model_name)
      changed1 = changed.replace('KUBEFLOW_NAMESPACE', KUBEFLOW_NAMESPACE)
      changed2 = changed1.replace('MODEL_PASSIN_VERSION', args.model_version)
      changed3 = changed2.replace('TRTSERVER_NAME', args.trtserver_name)
      changed4 = changed3.replace('WORKFLOW_NAME', args.workflow_name)
      changed5 = changed4.replace('WEBAPP_PREFIX', args.webapp_prefix)
      changed6 = changed5.replace('WEBAPP_PORT', args.webapp_port)
      target.write(changed6)


  logging.info('deploying web app.')
  subprocess.call(['kubectl', 'create', '-f', '/ml/trtis-clientapp.yaml'])


if __name__ == "__main__":
  main()
