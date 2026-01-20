import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from scipy import stats
from tabulate import tabulate
from collections import Counter
from typing import List, Dict, Generator
import gc


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=["../../LIME/", "../../Saliency/", "../../GradCAM/"],
    )
    parser.add_argument(
        "--method_names",
        nargs="+",
        default=["LIME", "Saliency", "GradCAM"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../analysis/",
    )
    parser.add_argument(
        "--top_points",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--top_overall_points",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="both",
        choices=["csv", "excel", "both"],
    )
    parser.add_argument(
        "--table_format",
        type=str,
        default="fancy_grid",
        choices=["grid", "simple", "fancy_grid", "github", "pipe", "orgtbl", "jira", "presto", "pretty", "psql", "rst", "mediawiki", "moinmoin", "youtrack", "html", "latex", "latex_raw", "latex_booktabs", "textile"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of files to load and process in memory at once"
    )
    return parser.parse_args()


def getDataFiles(data_dir: Path) -> List[Path]:
    """Get list of data files without loading them."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    data_files = list(data_dir.glob("*.pth"))
    if not data_files:
        raise FileNotFoundError(f"No .pth files found in {data_dir}")
    
    return sorted(data_files)


def loadDataInBatches(file_paths: List[Path], batch_size: int = 10) -> Generator[List[Dict], None, None]:
    """Load data files in batches to manage memory."""
    total_files = len(file_paths)
    
    for i in range(0, total_files, batch_size):
        batch_files = file_paths[i:i + batch_size]
        batch_data = []
        
        for file_path in batch_files:
            try:
                data = torch.load(file_path, map_location='cpu')
                batch_data.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        yield batch_data
        
        del batch_data
        gc.collect()


def calculateStatistics(explanation_tensor: torch.Tensor):
    flat_data = explanation_tensor.numpy().flatten()
    
    stats_dict = {
        'min': float(np.min(flat_data)),
        'max': float(np.max(flat_data)),
        'mean': float(np.mean(flat_data)),
        'std': float(np.std(flat_data)),
        'median': float(np.median(flat_data)),
        'q1': float(np.percentile(flat_data, 25)),
        'q3': float(np.percentile(flat_data, 75)),
        'iqr': float(np.percentile(flat_data, 75) - np.percentile(flat_data, 25)),
        'skewness': float(stats.skew(flat_data)),
        'kurtosis': float(stats.kurtosis(flat_data)),
        'positive_ratio': float(np.sum(flat_data > 0) / len(flat_data)),
        'negative_ratio': float(np.sum(flat_data < 0) / len(flat_data)),
        'zero_ratio': float(np.sum(flat_data == 0) / len(flat_data)),
        'abs_mean': float(np.mean(np.abs(flat_data))),
        'abs_std': float(np.std(np.abs(flat_data)))
    }
    
    return stats_dict


def findTopPoints(explanation_tensor: torch.Tensor, 
                  input_tensor: torch.Tensor,
                  top_n: int = 100):
    exp_np = explanation_tensor.numpy()
    abs_exp = np.abs(exp_np)
    
    flat_indices = np.argpartition(abs_exp.flatten(), -top_n)[-top_n:]
    top_indices = np.unravel_index(flat_indices, exp_np.shape)
    
    if exp_np.ndim == 5:
        batch_idx = top_indices[0][0] if exp_np.shape[0] == 1 else 0
        points = []
        
        for i in range(top_n):
            d, h, w = top_indices[2][i], top_indices[3][i], top_indices[4][i]
            point_info = {
                'coordinates': (int(d), int(h), int(w)),
                'explanation_value': float(exp_np[batch_idx, 0, d, h, w]),
                'absolute_value': float(abs_exp[batch_idx, 0, d, h, w]),
                'input_value': float(input_tensor[batch_idx, 0, d, h, w].numpy()) if input_tensor is not None else None
            }
            points.append(point_info)
    
    elif exp_np.ndim == 4:
        batch_idx = top_indices[0][0] if exp_np.shape[0] == 1 else 0
        points = []
        
        for i in range(top_n):
            d, h, w = top_indices[1][i], top_indices[2][i], top_indices[3][i]
            point_info = {
                'coordinates': (int(d), int(h), int(w)),
                'explanation_value': float(exp_np[batch_idx, d, h, w]),
                'absolute_value': float(abs_exp[batch_idx, d, h, w]),
                'input_value': float(input_tensor[batch_idx, d, h, w].numpy()) if input_tensor is not None else None
            }
            points.append(point_info)
    
    else:
        raise ValueError(f"Unsupported tensor dimension: {exp_np.ndim}")
    
    points.sort(key=lambda x: x['absolute_value'], reverse=True)
    
    return points


def processBatchStatistics(batch_data: List[Dict]) -> List[Dict]:
    """Process a batch of data for statistics."""
    batch_stats = []
    
    for i, sample_data in enumerate(batch_data):
        explanation_tensor = sample_data['explanation']
        sample_idx = sample_data.get('sample_idx', i)
        
        stats_dict = calculateStatistics(explanation_tensor)
        stats_dict['sample_idx'] = sample_idx
        batch_stats.append(stats_dict)
    
    return batch_stats


def processBatchTopPoints(batch_data: List[Dict], top_n: int) -> List[Dict]:
    """Process a batch of data for top points."""
    batch_top_points = []
    
    for i, sample_data in enumerate(batch_data):
        explanation_tensor = sample_data['explanation']
        input_tensor = sample_data.get('input', None)
        sample_idx = sample_data.get('sample_idx', i)
        
        try:
            top_points = findTopPoints(explanation_tensor, input_tensor, top_n)
            batch_top_points.append({
                'sample_idx': sample_idx,
                'top_points': top_points
            })
        except Exception as e:
            print(f"Warning: Could not extract top points for sample {sample_idx}: {e}")
    
    return batch_top_points


def aggregateStatistics(all_stats: List[Dict]) -> Dict:
    """Aggregate statistics from all samples."""
    if not all_stats:
        return {}
    
    stats_df = pd.DataFrame(all_stats)
    
    aggregated_stats = {}
    for col in stats_df.columns:
        if col not in ['sample_idx']:
            mean_val = stats_df[col].mean()
            
            if len(stats_df) > 1:
                std_val = stats_df[col].std()
                formatted_val = f"{mean_val:.6f} ± {std_val:.6f}"
            else:
                formatted_val = f"{mean_val:.6f}"
            
            aggregated_stats[col] = {
                'mean': mean_val,
                'std': std_val if len(stats_df) > 1 else None,
                'formatted': formatted_val,
                'has_std': len(stats_df) > 1
            }
    
    return aggregated_stats


def analyzeMethodStreaming(data_dir: Path, method_name: str, top_n: int = 100, 
                          batch_size: int = 10):
    """Streaming version of analyzeMethod that processes files in batches."""
    try:
        data_files = getDataFiles(data_dir)
    except FileNotFoundError as e:
        print(f"Error for {method_name}: {e}")
        return {
            'method_name': method_name,
            'data_dir': str(data_dir),
            'num_samples': 0,
            'num_analyzed': 0,
            'aggregated_statistics': {},
            'sample_statistics': [],
            'top_points': []
        }
    
    print(f"  Processing {len(data_files)} files for {method_name}...")
    
    all_stats = []
    all_top_points = []
    processed_files = 0
    
    for batch_data in loadDataInBatches(data_files, batch_size):
        batch_stats = processBatchStatistics(batch_data)
        all_stats.extend(batch_stats)
        
        batch_top_points = processBatchTopPoints(batch_data, top_n)
        all_top_points.extend(batch_top_points)
        
        processed_files += len(batch_data)
        
        if processed_files % 100 == 0:
            print(f"    Processed {processed_files}/{len(data_files)} files...")

        del batch_data
        gc.collect()

    aggregated_stats = aggregateStatistics(all_stats)
    
    results = {
        'method_name': method_name,
        'data_dir': str(data_dir),
        'num_samples': len(data_files),
        'num_analyzed': processed_files,
        'aggregated_statistics': aggregated_stats,
        'sample_statistics': all_stats,
        'top_points': all_top_points,
        'data_files': data_files
    }
    
    return results


def createComparisonTable(method_results):
    key_stats = ['min', 'max', 'mean', 'std', 'median', 'abs_mean']
    
    table_data = []
    headers = ['Method', 'Samples'] + [stat.upper() for stat in key_stats]
    
    for result in method_results:
        method_name = result['method_name']
        num_analyzed = result['num_analyzed']
        agg_stats = result.get('aggregated_statistics', {})
        
        row = [method_name, num_analyzed]
        for stat in key_stats:
            if stat in agg_stats:
                row.append(agg_stats[stat]['formatted'])
            else:
                row.append('N/A')
        
        table_data.append(row)
    
    return table_data, headers


def formatWithTabulate(table_data, headers, table_format="grid"):
    return tabulate(table_data, headers=headers, tablefmt=table_format, stralign="center", numalign="center")


def saveComparisonTable(table_data, headers, output_dir: Path, save_format: str, table_format: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    formatted_table = formatWithTabulate(table_data, headers, table_format)
    
    df = pd.DataFrame([row[2:] for row in table_data], 
                      columns=headers[2:],
                      index=[row[0] for row in table_data])
    
    if save_format in ['csv', 'both']:
        csv_path = output_dir / "comparison_table.csv"
        df.to_csv(csv_path)
    
    if save_format in ['excel', 'both']:
        excel_path = output_dir / "comparison_table.xlsx"
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Comparison')
            
            workbook = writer.book
            worksheet = writer.sheets['Comparison']
            
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1,
                'align': 'center'
            })
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)
            
            worksheet.set_column(0, 0, 15)
            worksheet.set_column(1, len(df.columns), 25)

    txt_path = output_dir / "comparison_table.txt"
    with open(txt_path, 'w') as f:
        f.write("XAI METHODS COMPARISON TABLE\n")
        f.write("="*80 + "\n")
        f.write(formatted_table)
        f.write("\n" + "="*80 + "\n")
    
    return df, formatted_table


def saveTopPoints(method_results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for result in method_results:
        method_name = result['method_name']
        top_points_data = result['top_points']
        
        if not top_points_data:
            continue
        
        json_data = []
        for sample_data in top_points_data:
            sample_entry = {
                'sample_idx': sample_data['sample_idx'],
                'top_points': sample_data['top_points']
            }
            json_data.append(sample_entry)
        
        json_file = output_dir / f"{method_name.lower()}_top_points.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)


def findTopOverallPointsByFrequency(method_results: List[Dict], top_overall_n: int = 50) -> Dict:
    method_overall_points = {}
    
    for result in method_results:
        method_name = result['method_name']
        top_points_data = result['top_points']
        
        if not top_points_data:
            method_overall_points[method_name] = []
            continue
        
        print(f"  Analyzing overall points for {method_name}...")
        
        point_counter = Counter()
        point_data = {}

        for sample_data in top_points_data:
            sample_idx = sample_data['sample_idx']
            
            for point in sample_data['top_points']:
                coords = point['coordinates']
                point_counter[coords] += 1
                
                if coords not in point_data:
                    point_data[coords] = {
                        'coordinates': coords,
                        'frequency': 0,
                        'max_absolute_value': -np.inf,
                        'max_explanation_value': -np.inf,
                        'total_absolute_value': 0.0,
                        'total_explanation_value': 0.0,
                        'samples': set()
                    }
                
                point_data[coords]['frequency'] += 1
                point_data[coords]['max_absolute_value'] = max(point_data[coords]['max_absolute_value'], point['absolute_value'])
                point_data[coords]['max_explanation_value'] = max(point_data[coords]['max_explanation_value'], point['explanation_value'])
                point_data[coords]['total_absolute_value'] += point['absolute_value']
                point_data[coords]['total_explanation_value'] += point['explanation_value']
                point_data[coords]['samples'].add(sample_idx)
        
        points_list = []
        for coords, data in point_data.items():
            data['avg_absolute_value'] = data['total_absolute_value'] / data['frequency']
            data['avg_explanation_value'] = data['total_explanation_value'] / data['frequency']
            data['num_samples'] = len(data['samples'])
            data['sample_indices'] = list(data['samples'])
            points_list.append(data)
        
        def sort_key(point):
            return (
                -point['frequency'],
                -point['max_explanation_value'],
                -point['max_absolute_value'],
                -point['avg_explanation_value']
            )
        
        points_list.sort(key=sort_key)
        
        method_overall_points[method_name] = points_list[:top_overall_n]
    
    return method_overall_points


def saveTopOverallPoints(overall_points: Dict, output_dir: Path, top_n: int = 50):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method_name, points in overall_points.items():
        if not points:
            continue
        
        print(f"  Saving overall points for {method_name}...")
        
        method_data = []
        for rank, point_info in enumerate(points, 1):
            coords = point_info['coordinates']
            
            method_data.append({
                'rank': rank,
                'coordinates_d': coords[0],
                'coordinates_h': coords[1],
                'coordinates_w': coords[2],
                'frequency': point_info['frequency'],
                'max_absolute_value': point_info['max_absolute_value'],
                'max_explanation_value': point_info['max_explanation_value'],
                'avg_absolute_value': point_info['avg_absolute_value'],
                'avg_explanation_value': point_info['avg_explanation_value'],
                'coords_str': f"({coords[0]},{coords[1]},{coords[2]})"
            })
        
        method_df = pd.DataFrame(method_data)
        
        csv_path = output_dir / f"overall_top_points_{method_name.lower()}.csv"
        method_df.to_csv(csv_path, index=False)
        
        txt_path = output_dir / f"overall_top_points_{method_name.lower()}_summary.txt"
        with open(txt_path, 'w') as f:
            f.write(f"OVERALL TOP POINTS - {method_name.upper()}\n")
            f.write("="*100 + "\n")
            f.write(f"Showing top {min(top_n, len(points))} most frequently occurring points across all samples\n")
            f.write("Frequency = Total number of occurrences in top N lists\n")
            f.write("Sorting: 1) Frequency (desc), 2) Max Explanation Value (desc), 3) Max Absolute Value (desc)\n")
            f.write("="*100 + "\n\n")
            
            table_data = []
            headers = ["Rank", "Coordinates", "Freq", "Max Exp", "Max Abs", "Avg Exp", "Avg Abs"]
            
            display_points = min(len(points), top_n)
            for rank, point_info in enumerate(points[:display_points], 1):
                coords = point_info['coordinates']
                
                table_data.append([
                    rank,
                    f"({coords[0]},{coords[1]},{coords[2]})",
                    point_info['frequency'],
                    f"{point_info['max_explanation_value']:.6f}",
                    f"{point_info['max_absolute_value']:.6f}",
                    f"{point_info['avg_explanation_value']:.6f}",
                    f"{point_info['avg_absolute_value']:.6f}"
                ])
            
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\n" + "="*100 + "\n")
            
            f.write("\nSummary Statistics:\n")
            f.write(f"Total unique points: {len(points)}\n")
            
            if points:
                frequencies = [p['frequency'] for p in points]
                max_exp_values = [p['max_explanation_value'] for p in points]
                avg_exp_values = [p['avg_explanation_value'] for p in points]
                
                f.write(f"Average frequency: {np.mean(frequencies):.2f}\n")
                f.write(f"Max frequency: {np.max(frequencies)}\n")
                f.write(f"Min frequency: {np.min(frequencies)}\n")
                f.write(f"Average max explanation value: {np.mean(max_exp_values):.6f}\n")
                f.write(f"Average explanation value: {np.mean(avg_exp_values):.6f}\n")
                
                positive_count = sum(1 for p in points if p['avg_explanation_value'] > 0)
                negative_count = sum(1 for p in points if p['avg_explanation_value'] < 0)
                
                f.write(f"Positive points: {positive_count} ({positive_count/len(points)*100:.1f}%)\n")
                f.write(f"Negative points: {negative_count} ({negative_count/len(points)*100:.1f}%)\n")
    
    return overall_points


def main():
    args = parseArguments()
    
    data_dirs = [Path(d) for d in args.data_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(data_dirs) != len(args.method_names):
        raise ValueError("Number of data directories must match number of method names")
    
    print(f"{'='*80}")
    print("XAI METHODS COMPARISON ANALYSIS (STREAMING VERSION)")
    print(f"{'='*80}")
    print(f"Methods to analyze: {', '.join(args.method_names)}")
    print(f"Data directories: {', '.join(str(d) for d in data_dirs)}")
    print(f"Output directory: {output_dir}")
    print(f"Top points per sample: {args.top_points}")
    print(f"Top overall points: {args.top_overall_points}")
    print(f"Batch size: {args.batch_size}")
    print(f"Table format: {args.table_format}")
    print(f"{'='*80}\n")
    
    all_results = []
    for data_dir, method_name in zip(data_dirs, args.method_names):
        print(f"\nAnalyzing {method_name}...")
        try:
            results = analyzeMethodStreaming(
                data_dir, 
                method_name, 
                args.top_points,
                args.batch_size
            )
            all_results.append(results)
            print(f"  Completed: {results['num_analyzed']} samples processed")
        except Exception as e:
            print(f"  Error analyzing {method_name}: {e}")
            all_results.append({
                'method_name': method_name,
                'data_dir': str(data_dir),
                'num_samples': 0,
                'num_analyzed': 0,
                'aggregated_statistics': {},
                'sample_statistics': [],
                'top_points': []
            })
    
    print(f"\n{'='*80}")
    print("CREATING COMPARISON TABLE")
    print(f"{'='*80}")
    table_data, headers = createComparisonTable(all_results)
    
    comparison_df, formatted_table = saveComparisonTable(
        table_data, headers, output_dir, args.save_format, args.table_format
    )
    
    print(formatted_table)
    
    # Save detailed statistics
    stats_dir = output_dir / "detailed_statistics"
    stats_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SAVING DETAILED STATISTICS")
    print(f"{'='*80}")
    
    for result in all_results:
        method_name = result['method_name']
        if result['sample_statistics']:
            stats_data = result['sample_statistics']
            stats_df = pd.DataFrame(stats_data)
            stats_file = stats_dir / f"{method_name.lower()}_detailed_stats.csv"
            stats_df.to_csv(stats_file, index=False)
            print(f"  {method_name}: {len(stats_data)} samples saved to {stats_file}")
    
    print(f"\n{'='*80}")
    print("SAVING TOP POINTS DATA")
    print(f"{'='*80}")
    
    top_points_dir = output_dir / "top_points"
    saveTopPoints(all_results, top_points_dir)
    
    print(f"\n{'='*80}")
    print("ANALYZING OVERALL TOP POINTS BY FREQUENCY")
    print(f"{'='*80}")
    
    overall_points = findTopOverallPointsByFrequency(all_results, args.top_overall_points)
    overall_points_dir = output_dir / "overall_points"
    saveTopOverallPoints(overall_points, overall_points_dir, args.top_overall_points)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved in: {output_dir}")
    
    print("\nKey Findings:")
    print("-" * 60)
    
    for result in all_results:
        if result['aggregated_statistics']:
            method_name = result['method_name']
            agg_stats = result['aggregated_statistics']
            num_samples = result['num_analyzed']
            
            print(f"\n{method_name} ({num_samples} sample{'s' if num_samples != 1 else ''}):")
            print(f"  Mean attribution: {agg_stats['mean']['formatted']}")
            print(f"  Absolute mean: {agg_stats['abs_mean']['formatted']}")
            print(f"  Min/Max: {agg_stats['min']['formatted']} / {agg_stats['max']['formatted']}")
    
    print(f"\nSummary:")
    print(f"- Total methods analyzed: {len([r for r in all_results if r['num_analyzed'] > 0])}")
    print(f"- Total samples processed: {sum(r['num_analyzed'] for r in all_results)}")
    print(f"- Results directory: {output_dir}")
    print(f"  ├── comparison_table.[csv/txt/xlsx]")
    print(f"  ├── detailed_statistics/")
    print(f"  ├── top_points/")
    print(f"  └── overall_points/")


if __name__ == "__main__":
    main()
