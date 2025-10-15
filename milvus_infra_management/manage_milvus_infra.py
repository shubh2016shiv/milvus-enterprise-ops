#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from typing import Dict, List
import time
from pathlib import Path

class MilvusInfraManager:
    def __init__(self):
        self.compose_file = Path(__file__).parent / "docker-compose.yml"
        self.services = {
            "etcd": {
                "name": "ETCD",
                "container": "milvus-etcd",
                "port": 2379,
                "description": "Metadata Storage"
            },
            "minio": {
                "name": "MinIO",
                "container": "milvus-minio",
                "port": 9000,
                "description": "Object Storage"
            },
            "standalone": {
                "name": "Milvus",
                "container": "milvus-standalone",
                "port": 19530,
                "description": "Vector Database"
            },
            "attu": {
                "name": "Attu",
                "container": "attu",
                "port": 8000,
                "description": "Web UI"
            }
        }

    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Execute a shell command and handle exceptions."""
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            sys.exit(1)

    def _check_docker(self):
        """Verify Docker is running."""
        try:
            self._run_command(["docker", "info"])
        except:
            print("Error: Docker is not running. Please start Docker and try again.")
            sys.exit(1)

    def _verify_compose_file(self):
        """Verify docker-compose.yml exists."""
        if not self.compose_file.exists():
            print(f"Error: {self.compose_file} not found.")
            sys.exit(1)

    def _get_container_status(self) -> Dict[str, bool]:
        """Get status of all containers."""
        result = {}
        for service in self.services:
            container_name = self.services[service]["container"]
            try:
                output = subprocess.run(
                    ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True
                )
                result[service] = bool(output.stdout.strip())
            except:
                result[service] = False
        return result
        
    def _check_existing_containers(self) -> Dict[str, bool]:
        """Check if containers exist (running or stopped)."""
        result = {}
        for service in self.services:
            container_name = self.services[service]["container"]
            try:
                output = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True
                )
                result[service] = bool(output.stdout.strip())
            except:
                result[service] = False
        return result

    def _print_status_table(self):
        """Print current status of services in markdown table format."""
        status = self._get_container_status()
        
        # Define column widths for better alignment
        col_widths = {
            "service": 10,
            "container": 18,
            "port": 8,
            "url": 25,
            "status": 10,
            "description": 20
        }
        
        print("\nMilvus Infrastructure Status:")
        header = (
            f"| {'Service'.ljust(col_widths['service'])} | "
            f"{'Container'.ljust(col_widths['container'])} | "
            f"{'Port'.ljust(col_widths['port'])} | "
            f"{'URL'.ljust(col_widths['url'])} | "
            f"{'Status'.ljust(col_widths['status'])} | "
            f"{'Description'.ljust(col_widths['description'])} |"
        )
        print(header)
        
        separator = (
            f"| {'-' * col_widths['service']} | "
            f"{'-' * col_widths['container']} | "
            f"{'-' * col_widths['port']} | "
            f"{'-' * col_widths['url']} | "
            f"{'-' * col_widths['status']} | "
            f"{'-' * col_widths['description']} |"
        )
        print(separator)
        
        # Get hostname - use localhost as default
        hostname = "localhost"
        
        for service_id, service_info in self.services.items():
            status_text = "Running" if status[service_id] else "Stopped"
            
            # Generate URL based on service
            if service_id == "attu":
                url = f"http://{hostname}:8000"
            elif service_id == "standalone":
                url = f"http://{hostname}:19530"
            elif service_id == "minio":
                url = f"http://{hostname}:9000"
            elif service_id == "etcd":
                url = f"http://{hostname}:2379"
            else:
                url = "N/A"
                
            row = (
                f"| {service_info['name'].ljust(col_widths['service'])} | "
                f"{service_info['container'].ljust(col_widths['container'])} | "
                f"{str(service_info['port']).ljust(col_widths['port'])} | "
                f"{url.ljust(col_widths['url'])} | "
                f"{status_text.ljust(col_widths['status'])} | "
                f"{service_info['description'].ljust(col_widths['description'])} |"
            )
            print(row)
        print()

    def _handle_existing_containers(self):
        """Handle existing containers before starting."""
        existing = self._check_existing_containers()
        
        if any(existing.values()):
            print("Found existing Milvus containers.")
            
            # Check if docker-compose project exists
            try:
                # Try to stop with docker-compose first (cleaner approach)
                print("Attempting to stop existing containers with docker-compose...")
                subprocess.run(
                    ["docker-compose", "-f", str(self.compose_file), "down"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False  # Don't raise exception if this fails
                )
                time.sleep(2)  # Give it time to clean up
                
                # Check if containers still exist
                still_existing = self._check_existing_containers()
                if any(still_existing.values()):
                    print("Some containers still exist. Removing them individually...")
                    for service, exists in still_existing.items():
                        if exists:
                            container_name = self.services[service]["container"]
                            print(f"Removing container: {container_name}")
                            subprocess.run(
                                ["docker", "rm", "-f", container_name],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=False
                            )
            except Exception as e:
                print(f"Warning: Error while handling existing containers: {str(e)}")
                print("Attempting to continue anyway...")
    
    def start(self):
        """Start Milvus infrastructure."""
        self._check_docker()
        self._verify_compose_file()
        
        # Handle existing containers to prevent conflicts
        self._handle_existing_containers()
        
        print("Starting Milvus infrastructure...")
        try:
            self._run_command(["docker-compose", "-f", str(self.compose_file), "up", "-d"])
            
            # Wait for services to be healthy
            print("Waiting for services to initialize...")
            time.sleep(5)
            print("\nMilvus infrastructure started successfully!")
            self._print_status_table()
        except Exception as e:
            print(f"Error starting Milvus infrastructure: {str(e)}")
            print("Try stopping any existing containers first with --stop")
            sys.exit(1)

    def stop(self):
        """Stop Milvus infrastructure."""
        self._verify_compose_file()
        self._check_docker()
        
        # Check if any containers are running
        existing = self._check_existing_containers()
        if not any(existing.values()):
            print("No Milvus containers found. Nothing to stop.")
            return
        
        print("Stopping Milvus infrastructure...")
        try:
            # Try docker-compose down first
            self._run_command(["docker-compose", "-f", str(self.compose_file), "down"])
            
            # Verify all containers are stopped
            still_existing = self._check_existing_containers()
            if any(still_existing.values()):
                print("Some containers still exist. Removing them individually...")
                for service, exists in still_existing.items():
                    if exists:
                        container_name = self.services[service]["container"]
                        print(f"Removing container: {container_name}")
                        subprocess.run(
                            ["docker", "rm", "-f", container_name],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False
                        )
            
            print("Milvus infrastructure stopped successfully!")
        except Exception as e:
            print(f"Error while stopping containers: {str(e)}")
            print("Attempting to force remove containers...")
            
            # Force remove all containers as last resort
            for service in self.services:
                container_name = self.services[service]["container"]
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False
                    )
                except:
                    pass
            
            print("Forced removal complete.")

    def restart(self):
        """Restart Milvus infrastructure."""
        self.stop()
        time.sleep(2)
        self.start()

def main():
    parser = argparse.ArgumentParser(description="Manage Milvus Infrastructure")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="Start Milvus infrastructure")
    group.add_argument("--stop", action="store_true", help="Stop Milvus infrastructure")
    group.add_argument("--restart", action="store_true", help="Restart Milvus infrastructure")

    args = parser.parse_args()
    manager = MilvusInfraManager()

    try:
        if args.start:
            manager.start()
        elif args.stop:
            manager.stop()
        elif args.restart:
            manager.restart()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
