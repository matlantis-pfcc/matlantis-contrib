# Matlantis Contrib

このリポジトリは[Matlantis](https://matlantis.com/ja/)を使う際に役に立つ便利なexampleを募集するcontribです. Matlantisのユーザーが作成したexampleを持ち寄ることで, より快適にMatlantisが利用可能となることを目指しています.

## Contributing
このcontribへexmapleを追加したい場合は[contributingガイドライン](CONTRIBUTING.md)に従ってPull Requestを作成してください.

## どのようなexampleを募集するか
Matlantisの活用を加速するexmapleとして例えば次のようなものが考えられます.
- 対象の原子系をわかりやすく可視化する
- Matlantisに実装されているfeatureの実行結果の集約, 分析を行う
- Matlantisに実装されているfeatureへの入力例を作る

上記のような内容のexampleの追加を歓迎します. また, これらはあくまでも一例であり, この他の内容についても便利なexmapleを作ることができたら是非このcontribへの追加をお願いします.

## How to use examples
募集したexampleは[matlantis_contrib_examples](matlantis_contrib_examples)に追加されていき, matlantis_contrib_examples内の各ディレクトリが1つのexampleに対応しています. 各exampleのディレクトリ構造は次のようになっています.

```
matlantis_contrib_examples
└── a_great_example(directory)
    ├── a_great_example.ipynb
    ├── input(directory)
    |   ├── hoge.xyz
    |   └── fuga.xyz
    └── output(directory)
        └── piyo.xyz
```
- `a_great_example`というexampleを実行するには`a_great_example`というディレクトリをzipファイルに圧縮してMatlantisにアップロードし, ファイルツリーのペインで右クリックから`Extract Archive`で解凍し、`a_great_example.ipynb`を実行します.
- exampleに入力ファイルが必要な場合は`a_great_example/input`に入力ファイルを置いてください.
- exampleが実行結果をファイルを出力する場合は `a_great_example/output`に結果が出力されます.


## 注意事項

* このcontribで提供するexampleの動作保証はいたしません。Matlantisの更新にともない動作しなくなる可能性があります。
* 今後、このcontribの内容を予告なく修正・削除することがあります。
