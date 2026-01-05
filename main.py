import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import os
import json
warnings.filterwarnings('ignore')

class ShillingAttackDetector:
    def __init__(self, time_interval_length=30):
        self.til = time_interval_length
        self.candidate_groups = []
        self.attack_groups = []
        self.suspicious_degrees = []
        self.data = None
        self.results = {}
        
    def generate_synthetic_data(self, n_users=2000, n_items=500, n_ratings=5000, 
                               n_attack_groups=5, attack_size_pct=10, filler_size_pct=2.5):
        print("\n[*] Generating synthetic dataset...")
        np.random.seed(42)
        
        genuine_users = np.arange(n_users)
        genuine_items = np.arange(n_items)
        base_date = datetime(2020, 1, 1)
        
        ratings_list = []
        for _ in range(n_ratings):
            user = np.random.choice(genuine_users)
            item = np.random.choice(genuine_items)
            rating = np.random.randint(1, 6)
            timestamp = base_date + timedelta(days=np.random.randint(0, 365))
            ratings_list.append({'user_id': user, 'item_id': item, 'rating': rating, 'timestamp': timestamp, 'is_attack': 0})
        
        attack_user_id = n_users
        for group_idx in range(n_attack_groups):
            target_item = np.random.choice(genuine_items)
            group_start_date = base_date + timedelta(days=np.random.randint(50, 300))
            attack_size = int(n_users * attack_size_pct / 100)
            
            for _ in range(attack_size):
                timestamp = group_start_date + timedelta(days=np.random.randint(0, 5))
                ratings_list.append({'user_id': attack_user_id, 'item_id': target_item, 'rating': 5, 'timestamp': timestamp, 'is_attack': 1})
                
                n_filler = int(n_items * filler_size_pct / 100)
                for _ in range(n_filler):
                    filler_item = np.random.choice(genuine_items)
                    ratings_list.append({'user_id': attack_user_id, 'item_id': filler_item, 'rating': np.random.randint(1, 6), 'timestamp': timestamp, 'is_attack': 1})
                
                attack_user_id += 1
        
        self.data = pd.DataFrame(ratings_list)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        print(f"✓ Dataset ready: {len(self.data)} ratings")
        os.makedirs('data', exist_ok=True)
        self.data.to_csv('data/synthetic_data.csv', index=False)
        return self.data
    
    def divide_candidate_groups(self):
        print("\n[*] Dividing candidate groups...")
        self.candidate_groups = []
        group_id = 0
        
        for item_id in self.data['item_id'].unique():
            item_ratings = self.data[self.data['item_id'] == item_id].copy()
            item_ratings = item_ratings.sort_values('timestamp').reset_index(drop=True)
            
            if len(item_ratings) == 0:
                continue
            
            start_idx = 0
            while start_idx < len(item_ratings):
                start_time = item_ratings.iloc[start_idx]['timestamp']
                end_time = start_time + timedelta(days=self.til)
                
                group_mask = (item_ratings['timestamp'] >= start_time) & (item_ratings['timestamp'] < end_time)
                group_users = item_ratings[group_mask]['user_id'].unique()
                
                if len(group_users) > 0:
                    self.candidate_groups.append({'group_id': group_id, 'item_id': item_id, 'users': set(group_users), 'start_time': start_time, 'end_time': end_time, 'num_users': len(group_users)})
                    group_id += 1
                
                next_idx = start_idx + len(item_ratings[group_mask])
                if next_idx >= len(item_ratings):
                    break
                start_idx = next_idx
        
        print(f"✓ Groups created: {len(self.candidate_groups)}")
        return self.candidate_groups
    
    def calculate_group_item_attention_degree(self, group):
        item_id = group['item_id']
        total_raters = len(self.data[self.data['item_id'] == item_id]['user_id'].unique())
        group_raters = len(group['users'])
        return group_raters / total_raters if total_raters > 0 else 0
    
    def calculate_user_activity(self, group):
        users_in_group = group['users']
        start_time = group['start_time']
        end_time = group['end_time']
        
        ua_values = []
        for user_id in users_in_group:
            user_data = self.data[self.data['user_id'] == user_id]
            total_ratings = len(user_data)
            if total_ratings == 0:
                continue
            sync_ratings = len(user_data[(user_data['timestamp'] >= start_time) & (user_data['timestamp'] < end_time)])
            ua = sync_ratings / total_ratings if total_ratings > 0 else 0
            ua_values.append(ua)
        
        return np.mean(ua_values) if ua_values else 0
    
    def calculate_suspicious_degrees(self):
        print("\n[*] Calculating suspicious degrees...")
        self.suspicious_degrees = []
        
        for group in self.candidate_groups:
            giad = self.calculate_group_item_attention_degree(group)
            ua = self.calculate_user_activity(group)
            suspicious_degree = (giad + ua) / 2
            
            self.suspicious_degrees.append({'group_id': group['group_id'], 'suspicious_degree': suspicious_degree, 'giad': giad, 'ua': ua, 'num_users': group['num_users']})
        
        print(f"✓ Degrees calculated: {len(self.suspicious_degrees)}")
        return self.suspicious_degrees
    
    def detect_attack_groups(self, eps=0.15, min_samples=2):
        print("\n[*] Detecting attack groups...")
        
        if not self.suspicious_degrees:
            return []
        
        sd_values = np.array([sd['suspicious_degree'] for sd in self.suspicious_degrees]).reshape(-1, 1)
        scaler = StandardScaler()
        sd_scaled = scaler.fit_transform(sd_values)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(sd_scaled)
        
        mean_sd = np.mean(sd_values)
        std_sd = np.std(sd_values)
        threshold = mean_sd + std_sd
        
        self.attack_groups = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_groups = [self.suspicious_degrees[i] for i in cluster_indices]
            cluster_mean_sd = np.mean([g['suspicious_degree'] for g in cluster_groups])
            
            if cluster_mean_sd >= threshold:
                for cg in cluster_groups:
                    self.attack_groups.append(cg)
        
        print(f"✓ Attack groups detected: {len(self.attack_groups)}")
        return self.attack_groups
    
    def evaluate(self, actual_column='is_attack'):
        print("\n[*] Evaluating performance...")
        
        if actual_column not in self.data.columns:
            return None
        
        attack_user_ids = set()
        for group in self.candidate_groups:
            if group['group_id'] in [ag['group_id'] for ag in self.attack_groups]:
                attack_user_ids.update(group['users'])
        
        y_true = self.data[actual_column].values
        y_pred = np.isin(self.data['user_id'].values, list(attack_user_ids)).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        self.results = {'precision': precision, 'recall': recall, 'f1': f1, 'true_positives': int(np.sum((y_true == 1) & (y_pred == 1))), 'false_positives': int(np.sum((y_true == 0) & (y_pred == 1))), 'true_negatives': int(np.sum((y_true == 0) & (y_pred == 0))), 'false_negatives': int(np.sum((y_true == 1) & (y_pred == 0)))}
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("="*50)
        
        return self.results
    
    def visualize_results(self):
        print("\n[*] Generating visualizations...")
        
        if not self.suspicious_degrees:
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = plt.subplot(2, 3, 1)
        sd_array = np.array([sd['suspicious_degree'] for sd in self.suspicious_degrees])
        ax1.hist(sd_array, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        ax1.set_title('Distribution of Suspicious Degrees')
        ax1.grid(alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        giad_array = np.array([sd['giad'] for sd in self.suspicious_degrees])
        ua_array = np.array([sd['ua'] for sd in self.suspicious_degrees])
        attack_indices = [ag['group_id'] for ag in self.attack_groups]
        colors = ['red' if self.suspicious_degrees[i]['group_id'] in attack_indices else 'blue' for i in range(len(self.suspicious_degrees))]
        ax2.scatter(giad_array, ua_array, c=colors, alpha=0.6, s=50)
        ax2.set_title('GIAD vs User Activity')
        ax2.grid(alpha=0.3)
        
        ax3 = plt.subplot(2, 3, 3)
        attack_count = len(self.attack_groups)
        normal_count = len(self.suspicious_degrees) - attack_count
        ax3.bar(['Attack', 'Normal'], [attack_count, normal_count], color=['red', 'blue'])
        ax3.set_title('Groups Classification')
        
        ax4 = plt.subplot(2, 3, 4)
        if self.results:
            metrics_values = [self.results['precision'], self.results['recall'], self.results['f1']]
            ax4.bar(['Precision', 'Recall', 'F1'], metrics_values, color=['green', 'orange', 'purple'])
            ax4.set_title('Evaluation Metrics')
            ax4.set_ylim([0, 1])
        
        ax5 = plt.subplot(2, 3, 5)
        group_sizes = [sd['num_users'] for sd in self.suspicious_degrees]
        ax5.hist(group_sizes, bins=20, edgecolor='black', color='lightgreen', alpha=0.7)
        ax5.set_title('Group Sizes Distribution')
        
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        ax6.text(0.5, 0.5, f'Total Groups: {len(self.suspicious_degrees)}\nAttack: {len(self.attack_groups)}\nDetection: {len(self.attack_groups)/len(self.suspicious_degrees)*100:.2f}%', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/detection_results.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/detection_results.png")
        plt.close()
    
    def save_report(self):
        print("\n[*] Saving report...")
        report = {'summary': {'total_groups': len(self.candidate_groups), 'attack_groups': len(self.attack_groups)}, 'metrics': self.results}
        os.makedirs('results', exist_ok=True)
        with open('results/report.json', 'w') as f:
            json.dump(report, f, indent=4, default=str)
        print("✓ Saved: results/report.json")

def main():
    print("\n" + "="*60)
    print("GROUP SHILLING ATTACK DETECTION SYSTEM")
    print("="*60)
    
    detector = ShillingAttackDetector(time_interval_length=30)
    detector.generate_synthetic_data()
    detector.divide_candidate_groups()
    detector.calculate_suspicious_degrees()
    detector.detect_attack_groups()
    detector.evaluate()
    detector.visualize_results()
    detector.save_report()
    
    print("\n" + "="*60)
    print("✓ DETECTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()