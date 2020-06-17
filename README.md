# TRTIS_GKE_kubeflow_pipeline_example

An example to run TRTIS server and client in a GKE cluster with kubeflow pipeline.

Upload the tar.gz file in kubeflow_tf folder, things should work as it is. 

1, set up local gcloud sdk, log in to gcp docker, set up kubeflow sdk in python

2, build containers in components/end2end-demo/contrainers,  components/kubeflow-tf/containers/trtis-serving-server

3, create kubeflow cluster in GCP

4, create GCS and persistent volumen in GKE (find online tutorial, search for names in the code), use kubectl -f foo.yaml (in https://yagr.nvidia.com/dongm/trtis_gke_kubeflow_pipeline_example/tree/master/components/kubeflow/persistent-pv-pvc) 

5, run python script in kubeflow-tf to get tar.gz file and upload to kubeflow cluster.

As a result of my sloppiness to maintain this code, user has to keep track of NGC tensorflow/trtis server/trtis client version when using updated NGC containers. The version need to be compatible as TRTIS server only accept certain tensorrt version. The rule of thumb is that NGC TF and TRITON/TRTIS should have the same version, for example 19.12
