atcoder環境
===

ubuntu@docker で atcoder-toolsを使ってc++17環境を作ります。
公式のatcoder-libraryも入ります

[環境構築]
* Atcoder用C++環境の用意 (Docker, Docker compose)
  * [docker環境を用意](https://docs.docker.jp/docker-for-windows/install.html)して、dockerとdocker-composeを使えるようにする
  * 下に置いてある、Dockerfile, docker-compose.ymlを同じフォルダに置く
    * テキストをコピペするか、ページ上部の「Downlowd ZIP」から取得する
  * Dockerfile、docker-compose.ymlを置いてあるフォルダで、powershellを開く(エクスプローラ>ファイル>powershellで開く)
  * powershell上で、「docker compose up -d」を実行する
* 開発環境の用意(VS Code)
  * [VSCodeをインストールする](https://azure.microsoft.com/ja-jp/products/visual-studio-code/)
* VSCodeからC++環境に接続する (Attach to runninc container)
  * VSCodeを開く
  * 左下端っこの緑の[><]みたいなマーク > Attach to running container 
  * atcoder用C++環境に接続できる
* 問題の取得 (agen abc***)
  * 左側エクスプローラから「フォルダーを開く」 > 「atcoder-workspace」
  * VScode上部で「ターミナル」を開く
  * 問題の取得 ターミナルで「agen abc001」を実行する
  * 何かをダウンロードしているログが出てきて、エクスプローラを見ると「abc001」のフォルダができている
* 実装環境への移動
  * ターミナルで、「cd abc001」「cd A」を実行する
  * エクスプローラで、「abc001」「A」「main.cpp」をクリックして、mainファイルを開く
  * [この問題](https://atcoder.jp/contests/abc001/tasks/abc001_1)を解く環境が開ける
* 実装
  * 「in_1.txt」「out_1.txt」が、テストケース。inを入力したらoutが出てくるような実装をする。
  * main.cppを適切に編集する
* テストする
  * ターミナルで「atest」を実行する
  * passしていたらOK
  * passしていなかったら、何かが間違っているので、直す
* 提出する (コピペ)
  * main.cppの内容を[Atcoderの問題ページ](https://atcoder.jp/contests/abc001/tasks/abc001_1)から提出する
    * ページ下部の「ソースコード」欄に提出する
* 提出する (API)
  * (最初の1回だけ)ターミナルで「agenlogin abc001」を実行する
  * Atcoderのパスワード・アカウントでログインする
  * 問題フォルダ(abc001/A)に移動して、ターミナルで「asub」を実行すると、main.cppが提出できる   

[コマンドエイリアス]
* 問題取得　agen : atcoder-tools --without-login gen (ex: agen abc002)
* ログインして問題取得　agenlogin : atcoder-tools gen
  * ログインして取得 最初はこれで問題を生成しておくと、submitが楽になります
* テスト　atest: atcoder-tools test 
  * 問題のディレクトリでatestすると、テストしてくれます
* 提出　asub: atcoder-tools submit
  * 問題のディレクトリでasubすると、テストしてから提出してくれます
  * -fすると、テスト結果によらずsubmitされます
    * 不定性のある問題(複数答えがあるうちの1つを構成、など)のときには-fをすると、気にせず提出できます

[TODO]
* gdbでデバッグできるようにする

[参考資料]
* 時間があれば、↓を読みましょう。本環境が利用しているライブラリです。
  * atocker-tools https://github.com/kyuridenamida/atcoder-tools
  * ac-library https://atcoder.github.io/ac-library/master/document_ja/
