import requests
import sys


class SparkJobSubmitter:
    def __init__(
        self,
        base_url="https://lab.samsai.io",
        token="f9318293d4e405e5cff5d03a348a02ae0c4331916cd390041a700045d1bcb16a",
    ):
        self.base_url = base_url
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def submit_job(
        self, python_file, student_id, cores=2, max_cores=4, memory="4g", priority="default"
    ):
        """Submit a job to the Spark cluster"""

        # Read code from file
        try:
            with open(python_file, "r") as file:
                code_content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{python_file}' not found")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

        # Prepare payload
        payload = {
            "student_id": student_id,
            "code": code_content,
            "cores": cores,
            "max_cores": max_cores,
            "memory": memory,
            "priority": priority,
        }

        # Submit job
        try:
            print(f"Submitting job for student {student_id}...")
            print(f"File: {python_file}")
            print(f"Resources: {cores} cores, {memory} memory")
            print(f"Priority: {priority}")
            print("-" * 50)

            response = requests.post(
                f"{self.base_url}/submit",
                headers=self.headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                print("✅ Job submitted successfully!")
                print(f"Job ID: {result['job_id']}")

                # Fixed: Use .get() with default values to prevent KeyError
                print(f"App Name: {result.get('app_name', 'Unknown')}")
                print(f"Queue Position: {result.get('queue_position', 'Unknown')}")
                print(f"Active Jobs: {result.get('active_jobs', 'Unknown')}")

                # Fixed: Check if status_url exists before accessing
                if "status_url" in result:
                    print(f"Status URL: {result['status_url']}")
                else:
                    print(f"Status URL: {self.base_url}/status/{result['job_id']}")

                if "log_url" in result:
                    print(f"Logs URL: {result['log_url']}")
                else:
                    print(
                        f"Logs URL: {self.base_url}/logs/{student_id}-{result['job_id']}.log"
                    )

                if "spark_ui_url" in result:
                    print(f"Spark UI: {result['spark_ui_url']}")
                else:
                    print(f"Spark UI: {self.base_url}/spark-ui")

                return result
            else:
                print(f"❌ Job submission failed!")
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
        except KeyError as e:
            print(f"❌ Missing key in response: {e}")
            print(
                f"Response received: {response.json() if 'response' in locals() else 'No response'}"
            )
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def get_job_status(self, job_id):
        """Get current status of a job"""
        try:
            response = requests.get(f"{self.base_url}/status/{job_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting status: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error getting job status: {e}")
            return None


    def get_cluster_status(self):
        """Get cluster status"""
        try:
            response = requests.get(f"{self.base_url}/cluster/status", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting cluster status: {e}")
        return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python submit.py <python_file> <student_id>")
        print("Example: python submit.py CH24M000.py CH24M000")
        sys.exit(1)

    python_file = sys.argv[1]
    student_id = sys.argv[2]
    cores = 1
    max_cores = 2
    memory = "1g"
    priority = "default"

    # Create submitter
    submitter = SparkJobSubmitter()

    # Show cluster status first
    print("🖥️  Cluster Status:")
    cluster_status = submitter.get_cluster_status()
    if cluster_status:
        resources = cluster_status.get("cluster_resources", {})
        apps = cluster_status.get("applications", {})
        print(
            f"   Available Cores: {resources.get('cores_available', 'Unknown')}/{resources.get('total_cores', 'Unknown')}"
        )
        print(f"   Active Jobs: {apps.get('active', 0)}")
        print(f"   Completed Jobs: {apps.get('completed', 0)}")
    print()

    # Submit job
    result = submitter.submit_job(python_file, student_id, cores, max_cores, memory, priority)

if __name__ == "__main__":
    main()
