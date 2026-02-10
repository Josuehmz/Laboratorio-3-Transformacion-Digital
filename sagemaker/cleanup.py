"""
Cleanup script to delete SageMaker resources and avoid charges
"""
import boto3
import argparse

def delete_endpoint(endpoint_name, region='us-east-1'):
    """Delete SageMaker endpoint and its configuration"""
    client = boto3.client('sagemaker', region_name=region)
    
    try:
        print(f'Deleting endpoint: {endpoint_name}...')
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f'✅ Endpoint deleted: {endpoint_name}')
        
        # Delete endpoint configuration
        try:
            print(f'Deleting endpoint configuration: {endpoint_name}...')
            client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            print(f'✅ Endpoint configuration deleted: {endpoint_name}')
        except client.exceptions.ClientError as e:
            if 'Could not find' in str(e):
                print(f'ℹ️  Endpoint configuration not found (may have been deleted)')
            else:
                print(f'⚠️  Error deleting endpoint configuration: {e}')
        
    except client.exceptions.ClientError as e:
        if 'Could not find' in str(e):
            print(f'ℹ️  Endpoint not found: {endpoint_name}')
        else:
            print(f'❌ Error deleting endpoint: {e}')
            raise

def list_endpoints(region='us-east-1'):
    """List all active SageMaker endpoints"""
    client = boto3.client('sagemaker', region_name=region)
    
    print('\n' + '='*60)
    print('Active SageMaker Endpoints')
    print('='*60)
    
    response = client.list_endpoints(StatusEquals='InService')
    endpoints = response['Endpoints']
    
    if not endpoints:
        print('No active endpoints found.')
    else:
        for i, endpoint in enumerate(endpoints, 1):
            print(f"\n{i}. {endpoint['EndpointName']}")
            print(f"   Status: {endpoint['EndpointStatus']}")
            print(f"   Created: {endpoint['CreationTime']}")
    
    print('='*60)
    return [ep['EndpointName'] for ep in endpoints]

def main():
    parser = argparse.ArgumentParser(description='Cleanup SageMaker resources')
    parser.add_argument('--endpoint-name', type=str, help='Specific endpoint to delete')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--list-only', action='store_true', help='Only list endpoints')
    parser.add_argument('--delete-all', action='store_true', help='Delete all endpoints')
    
    args = parser.parse_args()
    
    print('='*60)
    print('SageMaker Cleanup Tool')
    print('='*60)
    
    if args.list_only:
        list_endpoints(args.region)
        return
    
    if args.delete_all:
        endpoints = list_endpoints(args.region)
        if endpoints:
            confirm = input(f'\n⚠️  Delete {len(endpoints)} endpoint(s)? (yes/no): ')
            if confirm.lower() == 'yes':
                for endpoint in endpoints:
                    delete_endpoint(endpoint, args.region)
                print('\n✅ All endpoints deleted!')
            else:
                print('Cancelled.')
        return
    
    # Delete specific endpoint
    endpoint_name = args.endpoint_name
    
    # Try to read from file if not provided
    if not endpoint_name:
        try:
            with open('endpoint_name.txt', 'r') as f:
                endpoint_name = f.read().strip()
            print(f'Using endpoint from file: {endpoint_name}')
        except FileNotFoundError:
            print('Error: No endpoint name provided and endpoint_name.txt not found')
            print('\nOptions:')
            print('  1. python cleanup.py --endpoint-name YOUR_ENDPOINT_NAME')
            print('  2. python cleanup.py --list-only  (to see all endpoints)')
            print('  3. python cleanup.py --delete-all  (to delete all endpoints)')
            return
    
    confirm = input(f'\n⚠️  Delete endpoint "{endpoint_name}"? (yes/no): ')
    if confirm.lower() == 'yes':
        delete_endpoint(endpoint_name, args.region)
        print('\n✅ Cleanup complete!')
    else:
        print('Cancelled.')

if __name__ == '__main__':
    main()
