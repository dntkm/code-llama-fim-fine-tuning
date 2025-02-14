import requests
import os

# GitHub API配置
GITHUB_API = "https://api.github.com/search/repositories"
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]  # 替换为你的个人访问令牌
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}"  # 添加认证头
}
SEARCH_QUERY = "q=language:objective-c created:>2023-01-01"

# per page 100
def fetch_repos(max_page_count):
    page = 1
    all_repos = []
    while (page <= max_page_count):
        # 搜索仓库（带认证）
        response = requests.get(
            f"{GITHUB_API}?{SEARCH_QUERY}&page={page}&per_page=100",
            headers=HEADERS
        )
        # 处理API错误
        if response.status_code != 200:
            print(f"API请求失败，状态码：{response.status_code}, resp:{response.json()}")
            break

        repos = response.json()["items"]
        all_repos.extend(repos)
        print(f"get {len(repos)} repos")

        if "next" not in response.links:
            break
        else:
            page += 1

    print(f"get {len(all_repos)} repos in total")
    return all_repos

# return [(repo, count)]
def get_sorted_repos_stats(all_repos):
    repo_stats = []
    for i in range(1, len(all_repos)):
        progress = i / len(all_repos) * 100
        print("\r", end="")
        print("sort progress: {}%: ".format(progress), end="")

        repo = all_repos[i]
        tree_url = repo["trees_url"].replace("{/sha}", "/main?recursive=1")
        tree_response = requests.get(tree_url, headers=HEADERS)

        if tree_response.status_code == 200:
            files = tree_response.json().get("tree", [])
            objc_src_files = [f for f in files if f["path"].endswith((".m", ".mm"))]
            repo_stats.append( (repo, len(objc_src_files)) )
    # 按Objective-C文件数量降序
    return sorted(repo_stats, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    all_repos = fetch_repos(1)
    sorted_repos = get_sorted_repos_stats(all_repos)
    # 输出前20个仓库
    for repo, count in sorted_repos[:20]:
        print(f"仓库：{repo['full_name']}")
        print(f"创建时间：{repo['created_at']}")
        print(f"Objective-C文件数：{count}\n")
