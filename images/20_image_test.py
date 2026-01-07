"""
AWS Rekognition Integration Testing Suite
==========================================

Comprehensive evaluation script for testing AWS Rekognition preprocessing
in a Pix2Pix satellite-to-map translation pipeline.

This script tests three categories of images:
1. Qualified satellite imagery (should be processed)
2. Unqualified non-aerial content (should be filtered)
3. Edge cases (boundary conditions)

Generates detailed performance metrics including accuracy, latency, and cost analysis.

Author: Linh Nguyen
"""

import requests
import json
import base64
from PIL import Image
import io
import time
import os
from datetime import datetime

# API endpoint configuration
BASE_URL = "http://98.92.20.39:8000"

# Test suite configuration with expected outcomes
TEST_SUITE = {
    'qualified_satellite': {
        'images': ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', 
                   '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg'],
        'expected_status': 'success',
        'category': 'Qualified Satellite Imagery',
        'description': 'Standard satellite and aerial photographs'
    },
    'unqualified_non_aerial': {
        'images': ['ground_level.jpg', 'indoor.jpg', 'person.jpg'],
        'expected_status': 'filtered',
        'category': 'Non-Aerial Photography',
        'description': 'Ground-level, indoor, or portrait photographs'
    },
    'unqualified_low_quality': {
        'images': ['blank.jpg', 'noise.jpg', 'blurred_image.jpg'],
        'expected_status': 'filtered',
        'category': 'Low Quality Images',
        'description': 'Blank, noisy, or severely degraded images'
    },
    'edge_cases': {
        'images': ['existing_map.jpg', 'partially_cloudy.jpg', 
                   'mixed_terrain.jpg', 'rotated_image.png'],
        'expected_status': 'variable',
        'category': 'Edge Cases',
        'description': 'Boundary conditions and challenging scenarios'
    }
}

def test_single_image(image_path):
    """
    Test a single image through the Rekognition-enhanced pipeline.
    
    Args:
        image_path (str): Path to the test image
        
    Returns:
        dict: Test results including status, timing, and analysis data
    """
    
    if not os.path.exists(image_path):
        return {
            'success': False,
            'error': 'File not found',
            'image': image_path
        }
    
    try:
        start_time = time.time()
        
        # Send image to API endpoint
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/generate-enhanced",
                files={'image': f},
                timeout=60
            )
        
        total_time = time.time() - start_time
        
        if response.status_code != 200:
            return {
                'success': False,
                'error': f'HTTP {response.status_code}',
                'image': image_path,
                'response_time': total_time
            }
        
        result = response.json()
        
        # Enrich result with metadata
        result['image'] = image_path
        result['response_time'] = total_time
        result['success'] = True
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image': image_path
        }

def run_comprehensive_evaluation():
    """
    Execute complete evaluation across all test categories.
    
    Tests 20 images across qualified, unqualified, and edge case categories.
    Generates detailed performance metrics and cost analysis.
    
    Returns:
        dict: Comprehensive results including all metrics and categorized outcomes
    """
    
    print("="*80)
    print("AWS REKOGNITION INTEGRATION - COMPREHENSIVE EVALUATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize results structure
    all_results = {
        'processed_successfully': [],
        'filtered_correctly': [],
        'filtered_incorrectly': [],
        'accepted_incorrectly': [],
        'edge_case_results': [],
        'errors': [],
        'metadata': {
            'total_tested': 0,
            'test_date': datetime.now().isoformat(),
            'service_url': BASE_URL
        }
    }
    
    total_tested = 0
    
    # Iterate through each test category
    for suite_name, suite_config in TEST_SUITE.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {suite_config['category'].upper()}")
        print(f"Description: {suite_config['description']}")
        print(f"Expected Behavior: {suite_config['expected_status']}")
        print(f"{'='*80}")
        
        for img_path in suite_config['images']:
            total_tested += 1
            
            print(f"\n[{total_tested}/20] Testing: {img_path}")
            
            result = test_single_image(img_path)
            
            # Handle errors
            if not result['success']:
                print(f"   ✗ ERROR: {result.get('error', 'Unknown error')}")
                all_results['errors'].append(result)
                continue
            
            status = result['status']
            
            # Categorize results based on test category and outcome
            if suite_name == 'qualified_satellite':
                if status == 'success':
                    terrain = result['terrain_type']
                    confidence = result['confidence']
                    time_total = result['total_processing_time']
                    
                    print(f"   ✓ PROCESSED: {terrain.upper()} ({confidence:.1f}%) - {time_total:.2f}s")
                    
                    # Display top detected labels
                    for label in result['rekognition_analysis']['labels'][:3]:
                        print(f"      - {label['name']}: {label['confidence']:.1f}%")
                    
                    all_results['processed_successfully'].append(result)
                    
                elif status == 'filtered':
                    print(f"   ✗ FALSE NEGATIVE: {result['reason']}")
                    print(f"      Expected: Should have been processed")
                    all_results['filtered_incorrectly'].append(result)
            
            elif suite_name in ['unqualified_non_aerial', 'unqualified_low_quality']:
                if status == 'filtered':
                    reason = result['reason']
                    confidence = result.get('confidence', 0)
                    
                    print(f"   ✓ CORRECTLY FILTERED: {reason}")
                    print(f"      Confidence: {confidence:.1f}%")
                    
                    all_results['filtered_correctly'].append(result)
                    
                elif status == 'success':
                    terrain = result['terrain_type']
                    confidence = result['confidence']
                    
                    print(f"   ✗ FALSE POSITIVE: Incorrectly processed as {terrain.upper()}")
                    print(f"      Confidence: {confidence:.1f}%")
                    print(f"      Expected: Should have been rejected")
                    
                    all_results['accepted_incorrectly'].append(result)
            
            elif suite_name == 'edge_cases':
                # Edge cases have variable expected behavior - document actual outcome
                if status == 'success':
                    terrain = result['terrain_type']
                    confidence = result['confidence']
                    time_total = result['total_processing_time']
                    
                    print(f"   📊 PROCESSED: {terrain.upper()} ({confidence:.1f}%) - {time_total:.2f}s")
                    
                elif status == 'filtered':
                    reason = result['reason']
                    confidence = result.get('confidence', 0)
                    
                    print(f"   📊 FILTERED: {reason} (Confidence: {confidence:.1f}%)")
                
                # Display top labels for analysis
                if 'rekognition_analysis' in result:
                    for label in result['rekognition_analysis']['labels'][:3]:
                        print(f"      - {label['name']}: {label['confidence']:.1f}%")
                
                all_results['edge_case_results'].append(result)
            
            # Rate limiting - brief pause between requests
            time.sleep(0.5)
    
    # Update metadata
    all_results['metadata']['total_tested'] = total_tested
    
    # Generate comprehensive report
    generate_report(all_results)
    
    # Save raw results to JSON
    with open('comprehensive_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Raw results saved to: comprehensive_results.json")
    
    return all_results

def generate_report(results):
    """
    Generate detailed evaluation report with performance metrics.
    
    Analyzes accuracy, latency, costs, and provides actionable insights
    for the Rekognition integration performance.
    
    Args:
        results (dict): Complete test results from run_comprehensive_evaluation()
    """
    
    print("\n" + "="*80)
    print("PERFORMANCE EVALUATION REPORT")
    print("="*80)
    
    # Extract statistics
    total = results['metadata']['total_tested']
    processed = len(results['processed_successfully'])
    filtered_correct = len(results['filtered_correctly'])
    filtered_incorrect = len(results['filtered_incorrectly'])
    accepted_incorrect = len(results['accepted_incorrectly'])
    edge_cases = len(results['edge_case_results'])
    errors = len(results['errors'])
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total images tested: {total}")
    print(f"   Successfully processed: {processed}")
    print(f"   Correctly filtered: {filtered_correct}")
    print(f"   Incorrectly filtered: {filtered_incorrect}")
    print(f"   False positives: {accepted_incorrect}")
    print(f"   Edge cases: {edge_cases}")
    print(f"   Errors: {errors}")
    
    # Section 1: Classification Accuracy
    print(f"\n{'='*80}")
    print("1. CLASSIFICATION ACCURACY (Qualified Satellite Images)")
    print(f"{'='*80}")
    
    if processed > 0:
        # Confidence statistics
        confidences = [r['confidence'] for r in results['processed_successfully']]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        print(f"   Images processed: {processed}/10 expected")
        print(f"   Success rate: {(processed/10)*100:.1f}%")
        print(f"\n   Confidence Metrics:")
        print(f"      Average: {avg_conf:.2f}%")
        print(f"      Range: {min_conf:.1f}% - {max_conf:.1f}%")
        
        # Terrain classification distribution
        terrains = {}
        for r in results['processed_successfully']:
            t = r['terrain_type']
            terrains[t] = terrains.get(t, 0) + 1
        
        print(f"\n   Terrain Classification Distribution:")
        for terrain, count in sorted(terrains.items()):
            percent = (count / processed) * 100
            print(f"      {terrain.capitalize()}: {count} images ({percent:.1f}%)")
        
        # Latency analysis
        rekog_times = [r['rekognition_analysis']['processing_time'] 
                       for r in results['processed_successfully']]
        pix2pix_times = [r['pix2pix_processing_time'] 
                         for r in results['processed_successfully']]
        total_times = [r['total_processing_time'] 
                       for r in results['processed_successfully']]
        
        print(f"\n   Latency Breakdown:")
        print(f"      Rekognition: {sum(rekog_times)/len(rekog_times):.2f}s avg "
              f"(range: {min(rekog_times):.2f}s - {max(rekog_times):.2f}s)")
        print(f"      Pix2Pix: {sum(pix2pix_times)/len(pix2pix_times):.2f}s avg "
              f"(range: {min(pix2pix_times):.2f}s - {max(pix2pix_times):.2f}s)")
        print(f"      Total: {sum(total_times)/len(total_times):.2f}s avg "
              f"(range: {min(total_times):.2f}s - {max(total_times):.2f}s)")
        
        # Calculate standard deviation
        import statistics
        if len(total_times) > 1:
            sd = statistics.stdev(total_times)
            print(f"      Standard Deviation: {sd:.2f}s")
    
    # Section 2: Filtering Performance
    print(f"\n{'='*80}")
    print("2. FILTERING PERFORMANCE (Unqualified Images)")
    print(f"{'='*80}")
    
    total_unqualified = filtered_correct + accepted_incorrect
    
    if total_unqualified > 0:
        filter_accuracy = (filtered_correct / total_unqualified) * 100
        print(f"   Unqualified images tested: {total_unqualified}")
        print(f"   Correctly rejected: {filtered_correct}")
        print(f"   False positives: {accepted_incorrect}")
        print(f"   Filtering accuracy: {filter_accuracy:.1f}%")
        
        if filtered_correct > 0:
            print(f"\n   Rejection Reasons:")
            reasons = {}
            for r in results['filtered_correctly']:
                reason = r['reason']
                reasons[reason] = reasons.get(reason, 0) + 1
            
            for reason, count in reasons.items():
                print(f"      - {reason}: {count} images")
    
    # Section 3: Edge Case Analysis
    print(f"\n{'='*80}")
    print("3. EDGE CASE ANALYSIS")
    print(f"{'='*80}")
    
    if edge_cases > 0:
        print(f"   Edge cases tested: {edge_cases}")
        
        edge_processed = sum(1 for r in results['edge_case_results'] 
                            if r['status'] == 'success')
        edge_filtered = sum(1 for r in results['edge_case_results'] 
                           if r['status'] == 'filtered')
        
        print(f"   Processed: {edge_processed}")
        print(f"   Filtered: {edge_filtered}")
        
        print(f"\n   Individual Results:")
        for r in results['edge_case_results']:
            img = r['image']
            status = r['status']
            
            if status == 'success':
                terrain = r['terrain_type']
                conf = r['confidence']
                print(f"      ✓ {img}: Processed as {terrain.upper()} ({conf:.1f}%)")
            else:
                reason = r['reason']
                conf = r.get('confidence', 0)
                print(f"      ⊘ {img}: Rejected - {reason} ({conf:.1f}%)")
    
    # Section 4: Overall System Performance
    print(f"\n{'='*80}")
    print("4. OVERALL SYSTEM PERFORMANCE")
    print(f"{'='*80}")
    
    # Calculate overall accuracy
    correct_decisions = processed + filtered_correct
    incorrect_decisions = filtered_incorrect + accepted_incorrect
    total_decisive = correct_decisions + incorrect_decisions
    
    if total_decisive > 0:
        overall_accuracy = (correct_decisions / total_decisive) * 100
        print(f"   Total decisive test cases: {total_decisive}")
        print(f"   Correct decisions: {correct_decisions}")
        print(f"   Incorrect decisions: {incorrect_decisions}")
        print(f"   Overall system accuracy: {overall_accuracy:.1f}%")
    
    # Section 5: Cost Analysis
    print(f"\n{'='*80}")
    print("5. COST ANALYSIS")
    print(f"{'='*80}")
    
    total_processed = processed + edge_processed
    total_analyzed = results['metadata']['total_tested'] - errors
    
    # Cost calculations based on AWS pricing
    rekognition_cost = total_analyzed * 0.001  # $1 per 1,000 images
    pix2pix_cost = total_processed * 0.00052  # Only processed images
    total_cost = rekognition_cost + pix2pix_cost
    
    print(f"   Images analyzed by Rekognition: {total_analyzed}")
    print(f"   Images processed by Pix2Pix: {total_processed}")
    print(f"   Images filtered (saved processing cost): {filtered_correct}")
    
    print(f"\n   Cost Breakdown:")
    print(f"      Rekognition API: ${rekognition_cost:.6f}")
    print(f"      Pix2Pix Processing: ${pix2pix_cost:.6f}")
    print(f"      Total: ${total_cost:.6f}")
    
    if total_processed > 0:
        cost_per_processed = total_cost / total_processed
        cost_per_1000 = cost_per_processed * 1000
        print(f"\n   Per processed image: ${cost_per_processed:.6f}")
        print(f"   Per 1,000 processed: ${cost_per_1000:.2f}")
    
    # Cost savings from filtering
    if filtered_correct > 0:
        saved_cost = filtered_correct * 0.00052
        print(f"\n   Cost avoided by filtering: ${saved_cost:.6f}")
        print(f"      ({filtered_correct} unsuitable images prevented from processing)")
    
    # Section 6: Performance by Terrain Type
    print(f"\n{'='*80}")
    print("6. PERFORMANCE BY TERRAIN TYPE")
    print(f"{'='*80}")
    
    if processed > 0:
        # Group results by terrain classification
        by_terrain = {}
        for r in results['processed_successfully']:
            terrain = r['terrain_type']
            if terrain not by_terrain:
                by_terrain[terrain] = []
            by_terrain[terrain].append(r)
        
        for terrain, images in sorted(by_terrain.items()):
            print(f"\n   {terrain.upper()} ({len(images)} images):")
            
            # Confidence statistics for this terrain type
            confs = [img['confidence'] for img in images]
            avg_conf = sum(confs) / len(confs)
            
            print(f"      Average confidence: {avg_conf:.2f}%")
            print(f"      Confidence range: {min(confs):.1f}% - {max(confs):.1f}%")
            
            # Processing time statistics
            times = [img['total_processing_time'] for img in images]
            avg_time = sum(times) / len(times)
            print(f"      Average processing time: {avg_time:.2f}s")
            
            # Common labels for this terrain type
            all_labels = {}
            for img in images:
                for label in img['rekognition_analysis']['labels'][:5]:
                    name = label['name']
                    all_labels[name] = all_labels.get(name, [])
                    all_labels[name].append(label['confidence'])
            
            print(f"      Most common labels:")
            sorted_labels = sorted(all_labels.items(), 
                                  key=lambda x: sum(x[1])/len(x[1]), 
                                  reverse=True)
            for label_name, confs in sorted_labels[:3]:
                avg = sum(confs) / len(confs)
                print(f"         - {label_name}: {avg:.1f}% avg ({len(confs)} occurrences)")
    
    # Section 7: Key Findings Summary
    print(f"\n{'='*80}")
    print("7. KEY FINDINGS")
    print(f"{'='*80}")
    
    print(f"\n📈 ACCURACY METRICS:")
    if processed > 0:
        print(f"   ✓ Classification accuracy: {(processed/10)*100:.1f}% ({processed}/10 qualified images)")
    if total_unqualified > 0:
        print(f"   ✓ Filtering accuracy: {filter_accuracy:.1f}% ({filtered_correct}/{total_unqualified} unqualified images)")
    if total_decisive > 0:
        print(f"   ✓ Overall system accuracy: {overall_accuracy:.1f}% ({correct_decisions}/{total_decisive} correct decisions)")
    
    print(f"\n⏱️ LATENCY METRICS:")
    if processed > 0:
        print(f"   ✓ Mean processing time: {sum(total_times)/len(total_times):.2f}s")
        print(f"   ✓ Standard deviation: {sd:.2f}s")
        print(f"   ✓ Processing range: {min(total_times):.2f}s - {max(total_times):.2f}s")
        print(f"   ✓ Rekognition overhead: {(sum(rekog_times)/sum(total_times))*100:.1f}%")
    
    print(f"\n💰 COST METRICS:")
    if total_processed > 0:
        print(f"   ✓ Cost per 1,000 processed: ${cost_per_1000:.2f}")
        print(f"   ✓ Total cost for {total_analyzed} images: ${total_cost:.6f}")
    if filtered_correct > 0:
        print(f"   ✓ Cost savings from filtering: ${saved_cost:.6f}")
    
    print(f"\n🎯 CONFIDENCE DISTRIBUTION:")
    if processed > 0:
        ultra_high = sum(1 for c in confidences if c >= 99.5)
        high = sum(1 for c in confidences if 90 <= c < 99.5)
        medium = sum(1 for c in confidences if 70 <= c < 90)
        low = sum(1 for c in confidences if c < 70)
        
        print(f"   Ultra-high (≥99.5%): {ultra_high} ({(ultra_high/processed)*100:.1f}%)")
        print(f"   High (90-99.4%): {high} ({(high/processed)*100:.1f}%)")
        print(f"   Medium (70-89%): {medium} ({(medium/processed)*100:.1f}%)")
        print(f"   Low (<70%): {low} ({(low/processed)*100:.1f}%)")
    
    # Limitations and areas for improvement
    print(f"\n⚠️ LIMITATIONS IDENTIFIED:")
    if accepted_incorrect:
        print(f"   - {accepted_incorrect} false positive(s): Unsuitable images incorrectly accepted")
        for r in results['accepted_incorrectly']:
            print(f"      • {r['image']}: Classified as {r['terrain_type']} ({r['confidence']:.1f}%)")
            print(f"        Issue: Should have been rejected (low quality/non-aerial)")

    if filtered_incorrect:
        print(f"   - {filtered_incorrect} false negative(s): Suitable images incorrectly rejected")
        for r in results['filtered_incorrectly']:
            print(f"      • {r['image']}: {r['reason']}")
        
    if not accepted_incorrect and not filtered_incorrect:
        print(f"   ✅ No classification errors detected in test suite")
    
    # Edge case insights
    if edge_cases > 0:
        print(f"\n🔍 EDGE CASE INSIGHTS:")
        for r in results['edge_case_results']:
            img = r['image']
            status = r['status']
            
            if status == 'success':
                terrain = r['terrain_type']
                conf = r['confidence']
                print(f"   • {img}: Handled as {terrain} ({conf:.1f}%) - Demonstrates robustness")
            else:
                reason = r['reason']
                print(f"   • {img}: Rejected ({reason}) - Identifies system limitations")

def create_summary_table(results):
    """
    Generate formatted table of results suitable for documentation.
    
    Args:
        results (dict): Complete test results
    """
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY TABLE")
    print(f"{'='*80}")
    
    print(f"\n{'Image':<25} | {'Category':<20} | {'Status':<10} | {'Terrain':<10} | {'Conf %':<8} | {'Time':<8}")
    print("-"*100)
    
    # Successfully processed images
    for r in results['processed_successfully']:
        img = r['image']
        terrain = r['terrain_type']
        conf = f"{r['confidence']:.1f}"
        time_val = f"{r['total_processing_time']:.2f}s"
        print(f"{img:<25} | {'Qualified':<20} | {'Success':<10} | {terrain:<10} | {conf:<8} | {time_val:<8}")
    
    # Correctly filtered images
    for r in results['filtered_correctly']:
        img = r['image']
        conf = f"{r.get('confidence', 0):.1f}"
        print(f"{img:<25} | {'Unqualified':<20} | {'Filtered':<10} | {'-':<10} | {conf:<8} | {'-':<8}")
    
    # Edge case results
    for r in results['edge_case_results']:
        img = r['image']
        status = r['status']
        terrain = r.get('terrain_type', '-')
        conf = f"{r.get('confidence', 0):.1f}"
        time_val = f"{r.get('total_processing_time', 0):.2f}s" if status == 'success' else '-'
        print(f"{img:<25} | {'Edge Case':<20} | {status.capitalize():<10} | {terrain:<10} | {conf:<8} | {time_val:<8}")

if __name__ == '__main__':
    print("\n🚀 Starting comprehensive evaluation...")
    print("Testing 20 images across multiple categories\n")
    
    # Run evaluation
    results = run_comprehensive_evaluation()
    
    # Generate summary table
    create_summary_table(results)
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print("\nResults saved to: comprehensive_results.json")