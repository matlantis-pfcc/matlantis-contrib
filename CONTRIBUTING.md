# Contributing ガイドライン
このcontribではMatlantisの活用を加速するexampleを募集します. 想定するexampleは
- 対象の原子系をわかりやすく可視化する
- Matlantisに実装されているfeatureの実行結果の集約, 分析を行う
- Matlantisに実装されているfeatureへの入力例を作る

などです. これらは一例であり, この他のトピックについても幅広くexmapleを募集します. また, 原子シミュレーションを行っていく中で役に立つ内容であればnotebook上で必ずしもMatlantisをimportする必要はありません.

## contributeする前に
- このリポジトリのプログラムは[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)ライセンスの下で管理されます. このプロジェクトに参加するには[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)の内容を確認し, これに同意して頂くことが必要となります.
- [DCO](https://github.com/probot/dco#how-it-works)への署名が必要となります. `git commit`を行うときに`--signoff`または`-s`オプションを付与することでDCO署名を行うことが可能です.
- contribに追加されたexmapleの中で特に汎用性が高いものは[Matlantis](https://matlantis.com/ja/)のexampleやチュートリアルに追加されることがあります.
- contrib内のexmapleにバグなどがあることに気づいた場合は直接Pull Requestを送るのではなく, このリポジトリの[Issues](https://github.pfidev.jp/Matlantis/matlantis-contrib/issues)にissueを立てて頂くようお願いします.

## exampleの形式
exampleを追加するPull Requestを出す際には以下の内容が満たされているかに注意してください. また, Reviewerは主にこれらの内容が守られているかどうかを中心にReviewを行います.
- DCO署名がなされていること. DCO署名がなされているかどうかは[CI](https://github.com/probot/dco#how-it-works)(Continuous Integration)でチェックされます.
- 実行してエラーが出ないこと. Matlantis環境でnotebookのcellを上から順に実行していき, エラーが起きないことが必要です.
- コピーライト表記を含むこと. notebookの一番上にMarkdown形式で```Copyright xxx as contributors to Matlantis contrib project```という表記を追加してください. xxxにはプログラムの作成者の名前を入力します. [hello_world](https://github.pfidev.jp/Matlantis/matlantis-contrib/blob/master/matlantis_contrib_examples/hello_world/hello_world.ipynb)の内容を参考にしてください.
- 機密情報や認証情報(APIキーやパスワード)が含まれないこと.
