from retrieve import hybrid_retrieve, neighbor_expand_with_trace
from answer_with_llm import build_prompt
from scripts.all_llm import llm_gpt_4o_mini, llm_deepseek_v3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def call_llm(prompt: str) -> str:
    msg =  llm_deepseek_v3.invoke(prompt)
    return msg.content

def run(query: str):
    # 1) hybrid 检索命中
    hits = hybrid_retrieve(query, topk=25)

    # 2) 邻近扩展（并保留 trace：hits vs neighbors）
    hits, neighbors = neighbor_expand_with_trace(hits, window=2)

    # 3) 构造 prompt（显式分块展示 Top Hits / Neighbor Expansion）
    prompt = build_prompt(query, hits, neighbors)

    # 4) 调用 LLM
    answer = call_llm(prompt)

    return answer

if __name__ == "__main__":
    # q = "咏柳絮的《临江仙》是谁写的?"
    # q = "《秋窗风雨夕》是谁写的?"
    # q = "黛玉是什么样的性格?"
    # q = "关于黛玉的外貌描写在哪些段落?"
    # q = "关于王熙凤的外貌描写在哪些段落?"
    # q = "大观园里的女孩子中,谁最有文学才华?"
    # q = "绣春囊最可能是哪里来的?"
    # q = "袭人为什么想让宝玉搬出大观园?"
    # q = "薛宝钗人品怎么样?"
    # q = "第27章“金蝉脱壳”是怎么回事?体现了宝钗什么样的性格特点?"
    # q = "探春的居所什么样子的?体现了她什么样的性格特点?"
    # q = "王熙凤协理宁国府,管家严格,真的杜绝的所有的弊端吗?"
    # q = "晴雯为什么死了?"
    # q = "林黛玉进贾府时,她的外貌是怎样描写的?"
    # q = "林黛玉和晴雯有相似之处吗?"
    # q = "根据前 80 回的内容,推测贾府的最终命运"
    # q = "贾琏是一个怎样的人?"
    # q = "王熙凤为什么不孕?"
    # q = "已知麝香会导致不孕或流产,王熙凤咋就不孕?"
    # q = "夏金桂的性格和人品是不是很坏?"
    # q = "薛蟠的性格和人品是不是很坏?"
    # q = "四大家族中哪个家族教育水平最高?"
    # q = "薛宝钗是跟所有人都关系很好吗?"
    # q = "贾宝玉写的芙蓉女儿是指的谁?有暗喻吗?"
    # q = "秦可卿为什么就死了?"
    # q = "秦可卿说的当下的烈火烹油的事是什么?为什么这么说?"
    # q = "哪些地方体现了王熙凤的精明能干?"
    # q = "人造卫星有何用途?"
    # q = "林黛玉为什么喜欢李商隐的'留得残荷听雨声'这句诗?"
    # q = "雪融化的水都被谁喝过?为什么?"
    # q = "'冷香丸'是用什么做的?"
    # q = "书中有'反清复明'或者'悼明'的暗示吗?为什么?"
    q = "有个现代饭馆中有道菜叫'风流鹿肉',说是出自本书?"
    # q = "王熙凤做过什么亏心事?为什么?"
    print(run(q))
